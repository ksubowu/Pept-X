#!/usr/bin/env python3
"""Lariat peptide sequence -> SMILES.

Supported metadata includes ``|lariat``, ``|lariat_1`` and explicit residue
positions such as ``|lariat_3:2-5``. Explicit positions take precedence when
they satisfy the selected lariat chemistry. Invalid explicit positions fall
back deterministically to the compatible closure producing the largest ring.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import ast
import warnings

from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import rdmolops

from smi2seq_lariat import _canonical  # 复用规范化
from seq2smi import (seq2smi, N_CAPS, C_CAPS, anchors_and_leaving_from_helm,
                     _tokenize_preserve_brackets, _apply_custom_n_cap)  # 线性序列→SMILES 基础实现
from seq2smi_v2 import MonomerLib, build_fragment  # mapped monomer assembly
from utils import get_backbone_atoms

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LIB = BASE_DIR / "data" / "monomersFromHELMCoreLibrary.json"


LARIAT_RE = re.compile(
    r"(?:^|\|)(lariat(?:_([123]))?)(?::(\d+)\s*-\s*(\d+))?(?=\||$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LariatSpec:
    """Parsed lariat metadata; positions are one-based sequence positions."""

    kind: Optional[int] = None
    first: Optional[int] = None
    second: Optional[int] = None
    raw: Optional[str] = None

    @property
    def has_explicit_positions(self) -> bool:
        return self.first is not None and self.second is not None

    @property
    def type_tag(self) -> str:
        return f"lariat_{self.kind}" if self.kind else "lariat"

    @property
    def tag(self) -> str:
        if self.has_explicit_positions:
            return f"{self.type_tag}:{self.first}-{self.second}"
        return self.type_tag


def parse_lariat_spec(text: str) -> LariatSpec:
    """Parse ``|lariat[_n][:n1-n2]`` metadata from a sequence string."""

    match = LARIAT_RE.search(text or "")
    if not match:
        return LariatSpec()
    raw, kind_text, first_text, second_text = match.groups()
    return LariatSpec(
        kind=int(kind_text) if kind_text else None,
        first=int(first_text) if first_text else None,
        second=int(second_text) if second_text else None,
        raw=raw,
    )


def _parse_sequence(seq: str) -> Tuple[List[str], Optional[str]]:
    # 去掉 |lariat 标签
    core = seq.split("|")[0].strip()
    cap = None
    tokens: List[str] = []
    if core.startswith("[") and core.endswith("]"):
        try:
            parsed = ast.literal_eval(core)
            if isinstance(parsed, (list, tuple)):
                for item in parsed:
                    tok = str(item).strip().strip(".")
                    if not tok:
                        continue
                    if tok:
                        tokens.append(tok)
        except Exception:
            tokens = []
    if not tokens:
        tokens = _tokenize_preserve_brackets(core)
    if tokens and tokens[-1].lower().startswith("[c-cap:") and tokens[-1].endswith("]"):
        cap = tokens.pop()[7:-1].strip()
    return tokens, cap


def _strip_terminal_caps(tokens: List[str], lib: Optional[MonomerLib] = None) -> List[str]:
    if not tokens:
        return tokens
    core = list(tokens)
    head_code = (lib.resolve_code(core[0]) if lib else None) or core[0]
    head = head_code.lower()
    if head.startswith("n-cap:") or head in N_CAPS:
        core = core[1:]
    if not core:
        return core
    tail_code = (lib.resolve_code(core[-1]) if lib else None) or core[-1]
    tail = tail_code.lower()
    if tail.startswith("c-cap:") or tail in C_CAPS:
        core = core[:-1]
    return core


def _monomer_mol(lib: MonomerLib, code: str) -> Chem.Mol:
    entry = lib.get(code)
    smi = entry.get("smiles") or entry.get("smiles_L") or entry.get("smiles_D")
    if isinstance(smi, list):
        smi = next((s for s in smi if isinstance(s, str)), None)
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise ValueError(f"Invalid monomer SMILES for code {code}")
    return m


def _find_carbonyl_c(mol: Chem.Mol) -> int:
    # 使用 backbone 检测的 C
    bb = list(get_backbone_atoms(mol))
    if not bb:
        raise ValueError("Backbone not found in monomer")
    return bb[0][2]


def _find_amino_n(mol: Chem.Mol) -> int:
    bb = list(get_backbone_atoms(mol))
    if not bb:
        raise ValueError("Backbone not found in monomer")
    return bb[0][0]


def _strip_oh_from_c(rw: Chem.RWMol, c_idx: int) -> None:
    atom_c = rw.GetAtomWithIdx(c_idx)
    for nb in list(atom_c.GetNeighbors()):
        if nb.GetAtomicNum() == 8:
            bond = rw.GetBondBetweenAtoms(c_idx, nb.GetIdx())
            if bond.GetBondType() == rdchem.BondType.SINGLE:
                # 删除邻接的氢
                for h in list(nb.GetNeighbors()):
                    if h.GetAtomicNum() == 1:
                        rw.RemoveAtom(h.GetIdx())
                rw.RemoveAtom(nb.GetIdx())
                break


def _strip_h_from_n(rw: Chem.RWMol, n_idx: int) -> None:
    atom_n = rw.GetAtomWithIdx(n_idx)
    for nb in list(atom_n.GetNeighbors()):
        if nb.GetAtomicNum() == 1:
            rw.RemoveAtom(nb.GetIdx())
            break
    atom_n.SetNoImplicit(True)
    atom_n.SetNumExplicitHs(0)


def _ensure_amide_h(rw: Chem.RWMol, n_idx: int) -> None:
    """Guarantee amide-like N does not exceed valence 3 and has H if needed."""
    atom_n = rw.GetAtomWithIdx(n_idx)
    # Avoid valence queries (can trigger RDKit explicit-valence precondition).
    neighbor_count = len(atom_n.GetNeighbors())
    if neighbor_count < 3:
        h = Chem.Atom(1)
        h.SetNoImplicit(True)
        h_idx = rw.AddAtom(h)
        rw.AddBond(n_idx, h_idx, rdchem.BondType.SINGLE)
    atom_n.SetNoImplicit(True)
    atom_n.SetNumExplicitHs(0)


def _fix_overvalent_n(rw: Chem.RWMol) -> None:
    """Remove extra hydrogens from N atoms whose explicit valence exceeds 3."""
    for atom in list(rw.GetAtoms()):
        if atom.GetAtomicNum() != 7 or atom.GetIsAromatic():
            continue
        # neighbor count is more reliable here than cached explicit valence
        while len(atom.GetNeighbors()) > 3:
            h_nb = next((nb for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1), None)
            if h_nb is None:
                break
            rw.RemoveAtom(h_nb.GetIdx())
            atom = rw.GetAtomWithIdx(atom.GetIdx())
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)


def _attach_cap(base: Chem.Mol, c_idx: int, cap_smiles: str) -> Chem.Mol:
    cap_mol = Chem.MolFromSmiles(cap_smiles)
    if cap_mol is None:
        raise ValueError(f"Invalid C-cap SMILES: {cap_smiles}")
    combo = rdmolops.CombineMols(base, cap_mol)
    rw = Chem.RWMol(combo)
    c_global = c_idx
    # remove OH on terminal C before bonding
    _strip_oh_from_c(rw, c_global)

    # 重新定位 dummy（索引可能因删除而变化）
    dummy_global = next((a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0), None)
    if dummy_global is not None:
        neighbors = [nb.GetIdx() for nb in rw.GetAtomWithIdx(dummy_global).GetNeighbors()]
        for nb in neighbors:
            rw.AddBond(c_global, nb, rdchem.BondType.SINGLE)
        rw.RemoveAtom(dummy_global)
    else:
        # 没有占位符时，将 cap 的第一个原子与羧基碳成键
        offset = base.GetNumAtoms()
        rw.AddBond(c_global, offset, rdchem.BondType.SINGLE)
    return rw.GetMol()


def _find_sidechain_carbonyl(mol: Chem.Mol, residue: tuple) -> Optional[int]:
    ca_idx = residue[1]
    bb_atoms = set(residue)
    visited = set()
    stack = [ca_idx]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        atom = mol.GetAtomWithIdx(cur)
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx in bb_atoms:
                continue
            owner = mol.GetAtomWithIdx(ca_idx)
            if owner.HasProp("_lariat_residue") and (
                not nb.HasProp("_lariat_residue")
                or nb.GetIntProp("_lariat_residue") != owner.GetIntProp("_lariat_residue")
            ):
                continue
            # carbonyl C with double O and single O
            if nb.GetAtomicNum() == 6:
                bonds = nb.GetBonds()
                has_double_o = any(b.GetBondType() == rdchem.BondType.DOUBLE and b.GetOtherAtom(nb).GetAtomicNum() == 8 for b in bonds)
                has_single_o = any(b.GetBondType() == rdchem.BondType.SINGLE and b.GetOtherAtom(nb).GetAtomicNum() == 8 for b in bonds)
                if has_double_o and has_single_o:
                    return nb_idx
            stack.append(nb_idx)
    return None


def _find_sidechain_amine(mol: Chem.Mol, residue: tuple) -> Optional[int]:
    """Find a non-backbone nitrogen on the residue side-chain (e.g., Lys/Arg)."""
    ca_idx = residue[1]
    bb_atoms = set(residue)
    visited = set()
    stack = [ca_idx]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        atom = mol.GetAtomWithIdx(cur)
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx in bb_atoms:
                continue
            owner = mol.GetAtomWithIdx(ca_idx)
            if owner.HasProp("_lariat_residue") and (
                not nb.HasProp("_lariat_residue")
                or nb.GetIntProp("_lariat_residue") != owner.GetIntProp("_lariat_residue")
            ):
                continue
            if nb.GetAtomicNum() == 7:
                return nb_idx
            stack.append(nb_idx)
    return None


def _find_sidechain_hydroxyl(mol: Chem.Mol, residue: tuple) -> Optional[int]:
    """Find a non-backbone hydroxyl oxygen on the residue side-chain (e.g., Ser/Thr)."""
    ca_idx = residue[1]
    bb_atoms = set(residue)
    visited = set()
    stack = [ca_idx]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        atom = mol.GetAtomWithIdx(cur)
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx in bb_atoms:
                continue
            owner = mol.GetAtomWithIdx(ca_idx)
            if owner.HasProp("_lariat_residue") and (
                not nb.HasProp("_lariat_residue")
                or nb.GetIntProp("_lariat_residue") != owner.GetIntProp("_lariat_residue")
            ):
                continue
            if nb.GetAtomicNum() == 8 and nb.GetTotalNumHs(includeNeighbors=True) > 0:
                return nb_idx
            stack.append(nb_idx)
    return None


def _strip_h_from_o(rw: Chem.RWMol, o_idx: int) -> None:
    atom_o = rw.GetAtomWithIdx(o_idx)
    for nb in list(atom_o.GetNeighbors()):
        if nb.GetAtomicNum() == 1:
            rw.RemoveAtom(nb.GetIdx())
            break
    atom_o.SetNoImplicit(True)
    atom_o.SetNumExplicitHs(0)


def _base_code(token: str) -> str:
    t = token.strip()
    if t.lower().startswith("d-") and len(t) > 2:
        return t[2:]
    if t.lower().startswith("d") and len(t) > 1 and t[1].isalpha():
        return t[1:]
    if t.startswith("D-") and len(t) > 2:
        return t[2:]
    return t


def _find_atom_by_map(rw: Chem.RWMol, mapnum: int) -> Optional[int]:
    for atom in rw.GetAtoms():
        if atom.GetAtomMapNum() == mapnum:
            return atom.GetIdx()
    return None


def _explicit_closure(
    spec: LariatSpec,
    mol: Chem.Mol,
    matches: List[tuple],
) -> Tuple[Optional[dict], Optional[str]]:
    """Validate and resolve an explicit lariat closure."""

    if not spec.has_explicit_positions:
        return None, None
    if spec.kind not in (1, 2, 3):
        return None, "explicit residue positions require lariat_1, lariat_2, or lariat_3"
    first = spec.first
    second = spec.second
    count = len(matches)
    if first is None or second is None or not (1 <= first <= count and 1 <= second <= count):
        return None, f"positions must both be within 1..{count}"
    if first == second:
        return None, "the two attachment residues must be different"

    if spec.kind == 1:
        # lariat_1:n1-n2 = residue n1 backbone N to residue n2 side-chain COOH.
        if first != 1:
            return None, "lariat_1 backbone-N endpoint must be the free N-terminal residue (position 1)"
        atom_idx = _find_sidechain_carbonyl(mol, matches[second - 1])
        if atom_idx is None:
            return None, f"residue {second} has no compatible side-chain carboxyl carbon"
        return {
            "kind": 1,
            "mode": "sidechain_carbonyl",
            "closure_atom": atom_idx,
            "sidechain_res_idx": second - 1,
            "positions": [first, second],
        }, None

    # lariat_2/3:n1-n2 = residue n1 side chain to residue n2 backbone C.
    if second != count:
        return None, (
            f"lariat_{spec.kind} backbone-carbonyl endpoint must be the free "
            f"C-terminal residue (position {count})"
        )
    finder = _find_sidechain_amine if spec.kind == 2 else _find_sidechain_hydroxyl
    atom_idx = finder(mol, matches[first - 1])
    group = "amine nitrogen" if spec.kind == 2 else "hydroxyl oxygen"
    if atom_idx is None:
        return None, f"residue {first} has no compatible side-chain {group}"
    return {
        "kind": spec.kind,
        "mode": "sidechain_amine" if spec.kind == 2 else "sidechain_hydroxyl",
        "closure_atom": atom_idx,
        "sidechain_res_idx": first - 1,
        "positions": [first, second],
    }, None


def _default_closure(
    requested_kind: Optional[int],
    mol: Chem.Mol,
    matches: List[tuple],
) -> Optional[dict]:
    """Choose a compatible closure by type priority and maximum ring size."""

    count = len(matches)
    kinds = [requested_kind] if requested_kind in (1, 2, 3) else [1, 2, 3]
    for kind in kinds:
        if kind == 1:
            # N terminus is fixed; largest side-chain position gives largest ring.
            for idx in range(count - 1, 0, -1):
                atom_idx = _find_sidechain_carbonyl(mol, matches[idx])
                if atom_idx is not None:
                    return {
                        "kind": 1,
                        "mode": "sidechain_carbonyl",
                        "closure_atom": atom_idx,
                        "sidechain_res_idx": idx,
                        "positions": [1, idx + 1],
                    }
        elif kind == 2:
            # C terminus is fixed; smallest side-chain position gives largest ring.
            for idx in range(0, count - 1):
                atom_idx = _find_sidechain_amine(mol, matches[idx])
                if atom_idx is not None:
                    return {
                        "kind": 2,
                        "mode": "sidechain_amine",
                        "closure_atom": atom_idx,
                        "sidechain_res_idx": idx,
                        "positions": [idx + 1, count],
                    }
        else:
            for idx in range(0, count - 1):
                atom_idx = _find_sidechain_hydroxyl(mol, matches[idx])
                if atom_idx is not None:
                    return {
                        "kind": 3,
                        "mode": "sidechain_hydroxyl",
                        "closure_atom": atom_idx,
                        "sidechain_res_idx": idx,
                        "positions": [idx + 1, count],
                    }
    return None


def _assemble_lariat_backbone(tokens, lib):
    """Assemble mapped templates and retain residue identity through atom deletion.

    Caps are components, never residues. Backbone endpoints come from template
    R1/R2 anchors, so side-chain amides cannot introduce phantom residues.
    """
    components = []
    residue_count = 0
    tokens = list(tokens)
    custom_n = None
    if tokens and tokens[0].lower().startswith("[n-cap:") and tokens[0].endswith("]"):
        custom_n = tokens.pop(0)[7:-1].strip()
    for token in tokens:
        code = lib.resolve_code(token.strip("[]")) or token.strip("[]")
        entry = lib.get(code)
        fragment, anchors, _, _ = build_fragment(entry, False)
        role = str(entry.get("role", "")).lower()
        is_cap = role in {"ncap", "ccap"} or code.lower() in N_CAPS or code.lower() in C_CAPS
        # One-ended registered templates are caps even in older registries
        # lacking a role field.
        is_cap = is_cap or not ({1, 2} <= set(anchors))
        if not is_cap:
            residue_count += 1
            n, c = anchors[1], anchors[2]
            backbone = next((bb for bb in get_backbone_atoms(fragment)
                             if bb[0] == n and bb[2] == c), None)
            if backbone is None:
                neighbors = [a.GetIdx() for a in fragment.GetAtomWithIdx(n).GetNeighbors()
                             if a.GetAtomicNum() == 6 and a.GetIdx() != c]
                if len(neighbors) != 1:
                    raise ValueError(f"Cannot identify backbone in residue {residue_count} ({code}).")
                backbone = (n, neighbors[0], c)
            for atom in fragment.GetAtoms():
                atom.SetIntProp("_lariat_residue", residue_count)
            for label, idx in zip(("N", "CA", "C"), backbone):
                fragment.GetAtomWithIdx(idx).SetProp("_lariat_backbone", f"{residue_count}:{label}")
        components.append((fragment, anchors))
    if not components:
        raise ValueError("Empty peptide")
    mol, tail = components[0]
    consumed = []
    for fragment, anchors in components[1:]:
        if 2 not in tail or 1 not in anchors:
            raise ValueError("Adjacent components require R2-to-R1 backbone anchors.")
        offset = mol.GetNumAtoms()
        rw = Chem.RWMol(Chem.CombineMols(mol, fragment))
        left, right = tail[2], anchors[1] + offset
        for idx, number in ((left, 2), (right, 1)):
            leaving = [a.GetIdx() for a in rw.GetAtomWithIdx(idx).GetNeighbors()
                       if a.GetAtomMapNum() == number]
            if len(leaving) != 1:
                raise ValueError(f"Backbone R{number} leaving group is ambiguous.")
            consumed.extend(leaving)
        rw.AddBond(left, right, Chem.BondType.SINGLE)
        mol = rw.GetMol()
        tail = {number: idx + offset for number, idx in anchors.items()}
    rw = Chem.RWMol(mol)
    for idx in sorted(set(consumed), reverse=True):
        rw.RemoveAtom(idx)
    mol = rw.GetMol()
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.SanitizeMol(mol)
    if custom_n:
        mol = _apply_custom_n_cap(mol, custom_n)
        # The legacy cap helper sanitizes an N with implicit H disabled,
        # which can assign an artificial radical to a secondary amide.
        for atom in mol.GetAtoms():
            if atom.HasProp("_lariat_backbone") and atom.GetProp("_lariat_backbone") == "1:N":
                atom.SetNoImplicit(False)
                atom.SetNumRadicalElectrons(0)
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol)
    indices = {a.GetProp("_lariat_backbone"): a.GetIdx() for a in mol.GetAtoms()
               if a.HasProp("_lariat_backbone")}
    matches = [tuple(indices[f"{pos}:{label}"] for label in ("N", "CA", "C"))
               for pos in range(1, residue_count + 1)]
    return mol, matches


def build_smiles(
    seq: str,
    monomer_lib: Optional[MonomerLib] = None,
    lib_path: Path = DEFAULT_LIB,
    return_details: bool = False,
) -> Union[str, Tuple[str, dict]]:
    tokens, c_cap = _parse_sequence(seq)
    spec = parse_lariat_spec(seq)
    if not tokens:
        raise ValueError("Empty sequence")
    lib = monomer_lib if monomer_lib is not None else MonomerLib(str(lib_path))
    # Detect trailing cap token (e.g., -pip) when not provided as [C-cap:...]
    if tokens and not c_cap:
        tail_token = tokens[-1]
        resolved = lib.resolve_code(tail_token) or tail_token
        entry = lib.by_code.get(resolved.lower())
        if entry:
            smi = entry.get("smiles") or entry.get("smiles_L") or entry.get("smiles_D")
            if isinstance(smi, list):
                smi = next((s for s in smi if isinstance(s, str)), None)
            if smi:
                cap_mol = Chem.MolFromSmiles(smi)
                if cap_mol is not None and not list(get_backbone_atoms(cap_mol)):
                    c_cap = smi
                    tokens = tokens[:-1]

    mol, matches = _assemble_lariat_backbone(tokens, lib)
    if not matches:
        raise ValueError("Backbone not found when closing lariat.")

    head_n = matches[0][0]
    closure, fallback_reason = _explicit_closure(spec, mol, matches)
    fallback_used = spec.has_explicit_positions and closure is None
    if closure is None:
        closure = _default_closure(spec.kind, mol, matches)
    if closure is None:
        suffix = f" for {spec.type_tag}" if spec.kind else ""
        reason = f" Explicit request invalid: {fallback_reason}." if fallback_reason else ""
        raise ValueError(f"No compatible lariat closure site found{suffix}.{reason}")
    if fallback_used:
        warnings.warn(
            f"Invalid explicit lariat specification '{spec.tag}': {fallback_reason}; "
            f"using maximum-ring fallback {closure['positions'][0]}-{closure['positions'][1]}.",
            RuntimeWarning,
            stacklevel=2,
        )

    closure_mode = closure["mode"]
    closure_atom = closure["closure_atom"]
    closure_res_idx = closure["sidechain_res_idx"]

    if c_cap and closure_mode != "sidechain_carbonyl":
        raise ValueError("C-cap present but no D/E sidechain closure site found for lariat.")

    rw = Chem.RWMol(mol)
    tail_res_idx = len(matches) - 1
    map_head = 9001
    map_tail = 9002
    map_closure = 9003
    # Preserve the actual terminal backbone carbonyl before closing the lariat.
    # Ring closure can change RDKit's get_backbone_atoms() traversal order, so
    # a later numeric residue index may point at the wrong carbonyl.
    ccap_target_marked = False
    if c_cap:
        rw.GetAtomWithIdx(matches[-1][2]).SetAtomMapNum(map_tail)
        ccap_target_marked = True

    if closure_mode == "sidechain_carbonyl":
        rw.GetAtomWithIdx(head_n).SetAtomMapNum(map_head)
        rw.GetAtomWithIdx(closure_atom).SetAtomMapNum(map_closure)
        head_idx = _find_atom_by_map(rw, map_head)
        clos_idx = _find_atom_by_map(rw, map_closure)
        _strip_h_from_n(rw, head_idx)
        head_idx = _find_atom_by_map(rw, map_head)
        clos_idx = _find_atom_by_map(rw, map_closure)
        rw.AddBond(head_idx, clos_idx, rdchem.BondType.SINGLE)
        _strip_oh_from_c(rw, clos_idx)
        head_idx = _find_atom_by_map(rw, map_head)
        _ensure_amide_h(rw, head_idx)
    else:
        tail_c = matches[tail_res_idx][2]
        rw.GetAtomWithIdx(tail_c).SetAtomMapNum(map_tail)
        rw.GetAtomWithIdx(closure_atom).SetAtomMapNum(map_closure)
        tail_idx = _find_atom_by_map(rw, map_tail)
        clos_idx = _find_atom_by_map(rw, map_closure)
        if closure_mode == "sidechain_amine":
            _strip_h_from_n(rw, clos_idx)
        else:
            _strip_h_from_o(rw, clos_idx)
        tail_idx = _find_atom_by_map(rw, map_tail)
        clos_idx = _find_atom_by_map(rw, map_closure)
        rw.AddBond(tail_idx, clos_idx, rdchem.BondType.SINGLE)
        tail_idx = _find_atom_by_map(rw, map_tail)
        _strip_oh_from_c(rw, tail_idx)

    for atom in rw.GetAtoms():
        atom_map = atom.GetAtomMapNum()
        if atom_map in (map_head, map_closure) or (atom_map == map_tail and not ccap_target_marked):
            atom.SetAtomMapNum(0)

    # 需要 C-cap 时，在末端主链羧基碳上拼接（去掉占位符）
    if c_cap:
        c_idx = _find_atom_by_map(rw, map_tail)
        if c_idx is None:
            raise ValueError("Marked terminal backbone carbonyl not found when attaching C-cap.")
        cap_mol = Chem.MolFromSmiles(c_cap, sanitize=False)
        skip_carboxyl = False
        if cap_mol is not None:
            # If cap has HELM anchor map numbers, attach via anchor directly.
            anchor_atoms = [a for a in cap_mol.GetAtoms() if a.GetAtomMapNum() > 0]
            if anchor_atoms:
                # Build cap by HELM anchors: remove leaving atom (e.g., [H:1]) then bond anchor heavy atom.
                try:
                    anchors, leaving = anchors_and_leaving_from_helm(cap_mol)
                    cap_rw = Chem.RWMol(cap_mol)
                    for atom in cap_rw.GetAtoms():
                        atom.SetIntProp("_orig_idx", atom.GetIdx())
                    removed = sorted(leaving, reverse=True)
                    for idx in removed:
                        if idx < cap_rw.GetNumAtoms():
                            cap_rw.RemoveAtom(idx)
                    cap_clean = cap_rw.GetMol()
                    try:
                        Chem.SanitizeMol(cap_clean)
                    except Exception:
                        pass
                    # adjust anchor index after removals using original index mapping
                    anchor_orig = anchors.get(1, anchor_atoms[0].GetIdx())
                    # indices shift down for each removed atom with idx < anchor_orig
                    anchor_idx = anchor_orig
                    for idx in removed:
                        if idx < anchor_orig:
                            anchor_idx -= 1
                    if anchor_idx < 0 or anchor_idx >= cap_clean.GetNumAtoms():
                        anchor_idx = 0
                    combo = rdmolops.CombineMols(rw.GetMol(), cap_clean)
                    rw_cap = Chem.RWMol(combo)
                    offset = rw.GetMol().GetNumAtoms()
                    anchor_idx = offset + anchor_idx
                    rw_cap.AddBond(c_idx, anchor_idx, rdchem.BondType.SINGLE)
                    # After bonding, remove OH on the carbonyl carbon and any explicit H on anchor.
                    _strip_oh_from_c(rw_cap, c_idx)
                    for nb in list(rw_cap.GetAtomWithIdx(anchor_idx).GetNeighbors()):
                        if nb.GetAtomicNum() == 1:
                            rw_cap.RemoveAtom(nb.GetIdx())
                            break
                    for atom in rw_cap.GetAtoms():
                        if atom.GetAtomMapNum() > 0:
                            atom.SetAtomMapNum(0)
                    rw = rw_cap
                    cap_mol = None
                    skip_carboxyl = True
                except Exception:
                    pass
            if cap_mol is not None:
                # Fallback for caps without anchors: use single ring N as anchor (e.g., -pip)
                n_atoms = [a for a in cap_mol.GetAtoms() if a.GetSymbol() == "N"]
                has_carbonyl = any(
                    a.GetSymbol() == "C"
                    and any(
                        b.GetBondType() == rdchem.BondType.DOUBLE
                        and b.GetOtherAtom(a).GetAtomicNum() == 8
                        for b in a.GetBonds()
                    )
                    for a in cap_mol.GetAtoms()
                )
                if len(n_atoms) == 1 and not has_carbonyl:
                    combo = rdmolops.CombineMols(rw.GetMol(), cap_mol)
                    rw_cap = Chem.RWMol(combo)
                    offset = rw.GetMol().GetNumAtoms()
                    anchor_idx = offset + n_atoms[0].GetIdx()
                    rw_cap.AddBond(c_idx, anchor_idx, rdchem.BondType.SINGLE)
                    _strip_oh_from_c(rw_cap, c_idx)
                    for nb in list(rw_cap.GetAtomWithIdx(anchor_idx).GetNeighbors()):
                        if nb.GetAtomicNum() == 1:
                            rw_cap.RemoveAtom(nb.GetIdx())
                            break
                    rw = rw_cap
                    cap_mol = None
                    skip_carboxyl = True
            if cap_mol is not None:
                try:
                    Chem.SanitizeMol(cap_mol)
                except Exception:
                    pass
                for atom in cap_mol.GetAtoms():
                    if atom.GetAtomicNum() != 6:
                        continue
                    has_double_o = any(
                        b.GetBondType() == rdchem.BondType.DOUBLE and b.GetOtherAtom(atom).GetAtomicNum() == 8
                        for b in atom.GetBonds()
                    )
                    has_single_o = any(
                        b.GetBondType() == rdchem.BondType.SINGLE and b.GetOtherAtom(atom).GetAtomicNum() == 8
                        for b in atom.GetBonds()
                    )
                    if has_double_o and has_single_o:
                        skip_carboxyl = True
                        break
        if not skip_carboxyl:
            rw = Chem.RWMol(_attach_cap(rw.GetMol(), c_idx, c_cap))

    for atom in rw.GetAtoms():
        if atom.GetAtomMapNum() > 0:
            atom.SetAtomMapNum(0)

    _fix_overvalent_n(rw)
    final_mol = rw.GetMol()
    Chem.SanitizeMol(final_mol)
    final_smiles = Chem.MolToSmiles(final_mol, isomericSmiles=True)
    details: Dict[str, object] = {
        "requested_lariat": spec.tag if spec.raw else "lariat",
        "explicit_positions_requested": spec.has_explicit_positions,
        "explicit_positions_valid": spec.has_explicit_positions and not fallback_used,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason if fallback_used else None,
        "selection_rule": "explicit_positions" if spec.has_explicit_positions and not fallback_used else "maximum_ring",
        "selected_lariat": f"lariat_{closure['kind']}:{closure['positions'][0]}-{closure['positions'][1]}",
        "lariat_type": f"lariat_{closure['kind']}",
        "lariat_positions": closure["positions"],
        "closure_mode": closure_mode,
    }
    return (final_smiles, details) if return_details else final_smiles


def main():
    parser = argparse.ArgumentParser(description="Lariat sequence -> SMILES")
    parser.add_argument("--input", type=Path, default=Path("seq2smi_lariat_input.txt"))
    parser.add_argument("--output", type=Path, default=Path("seq2smi_lariat_out.txt"))
    parser.add_argument("--lib", type=Path, default=DEFAULT_LIB)
    args = parser.parse_args()

    lines = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        seq = line.strip()
        if not seq:
            continue
        try:
            smi = build_smiles(seq, lib_path=args.lib)
            lines.append(f"{seq}\t{smi}")
        except Exception as exc:  # pylint: disable=broad-except
            lines.append(f"{seq}\tERROR: {exc}")
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
