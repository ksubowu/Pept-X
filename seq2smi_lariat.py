#!/usr/bin/env python3
"""Lariat peptide sequence -> SMILES (approximate linear reconstruction).

输入格式：A-A-L-...-D-[C-cap:*N1CCCCC1]|lariat
当前实现：
- 解析核心序列与可选 C-cap（占位符 * 连接终末羧基碳）；
- 使用单体库酸式 SMILES 构建线性肽链（仅主链肽键）；
- 若提供 C-cap，则用占位符 * 与末端羧基碳成键；
- 暂不重建套索环的异肽键（缺少位点信息），输出为线性加 C-cap 的 SMILES。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple
import ast

from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import rdmolops

from smi2seq_lariat import _canonical  # 复用规范化
from seq2smi import seq2smi, N_CAPS, C_CAPS, anchors_and_leaving_from_helm  # 线性序列→SMILES 基础实现
from seq2smi_v2 import MonomerLib  # 复用单体库定义
from utils import get_backbone_atoms

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LIB = BASE_DIR / "data" / "monomersFromHELMCoreLibrary.json"


def _parse_sequence(seq: str) -> Tuple[List[str], Optional[str]]:
    # 去掉 |lariat 标签
    core = seq.split("|")[0].strip()
    cap = None
    m = re.search(r"\[C-cap:([^\]]+)\]", core)
    if m:
        cap = m.group(1)
        core = core.replace(m.group(0), "")
    tokens: List[str] = []
    if core.startswith("[") and core.endswith("]"):
        try:
            parsed = ast.literal_eval(core)
            if isinstance(parsed, (list, tuple)):
                for item in parsed:
                    tok = str(item).strip()
                    if not tok:
                        continue
                    tok = tok.strip(".").strip("-")
                    if tok:
                        tokens.append(tok)
        except Exception:
            tokens = []
    if not tokens:
        sep = "." if "." in core else "-"
        tokens = [tok.strip().strip(".").strip("-") for tok in core.split(sep) if tok.strip()]
    return tokens, cap


def _strip_terminal_caps(tokens: List[str]) -> List[str]:
    if not tokens:
        return tokens
    core = list(tokens)
    head = core[0].lower()
    if head.startswith("n-cap:") or head in N_CAPS:
        core = core[1:]
    if not core:
        return core
    tail = core[-1].lower()
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
        if atom.GetAtomicNum() != 7:
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
            if nb.GetAtomicNum() == 8 and nb.GetTotalNumHs() > 0:
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


def build_smiles(seq: str, monomer_lib: Optional[MonomerLib] = None, lib_path: Path = DEFAULT_LIB) -> str:
    tokens, c_cap = _parse_sequence(seq)
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
    core_tokens = _strip_terminal_caps(tokens)

    # 先用成熟的线性转换生成主链，避免手写拼接导致价态错误
    base_sequence = ".".join(tokens)
    base_smiles = seq2smi(base_sequence, lib)
    mol = Chem.MolFromSmiles(base_smiles)
    if mol is None:
        raise ValueError("Failed to build base linear peptide from sequence.")

    matches = list(get_backbone_atoms(mol))
    if not matches:
        raise ValueError("Backbone not found when closing lariat.")
    if len(matches) != len(core_tokens):
        raise ValueError("Residue count mismatch after cap parsing.")

    head_n = matches[0][0]
    closure_mode = "none"
    closure_atom = None
    closure_res_idx = None

    def _pick_indices(targets: set, reverse: bool = False) -> list:
        ordered = list(range(len(core_tokens)))
        if reverse:
            ordered.reverse()
        return [idx for idx in ordered if _base_code(core_tokens[idx]).upper() in targets]

    # Priority 1: D/E sidechain COOH -> head backbone N.
    for idx in _pick_indices({"D", "E"}, reverse=True):
        side_c = _find_sidechain_carbonyl(mol, matches[idx])
        if side_c is not None:
            closure_mode = "sidechain_carbonyl"
            closure_atom = side_c
            closure_res_idx = idx
            break

    # Priority 2: tail backbone COOH -> K/R sidechain amine.
    if closure_mode == "none":
        for idx in _pick_indices({"K", "R"}):
            side_n = _find_sidechain_amine(mol, matches[idx])
            if side_n is not None:
                closure_mode = "sidechain_amine"
                closure_atom = side_n
                closure_res_idx = idx
                break

    # Priority 3: tail backbone COOH -> S/T sidechain hydroxyl (ester).
    if closure_mode == "none":
        for idx in _pick_indices({"S", "T"}):
            side_o = _find_sidechain_hydroxyl(mol, matches[idx])
            if side_o is not None:
                closure_mode = "sidechain_hydroxyl"
                closure_atom = side_o
                closure_res_idx = idx
                break

    if closure_mode == "none":
        raise ValueError("No lariat closure site found for this sequence.")

    if c_cap and closure_mode != "sidechain_carbonyl":
        raise ValueError("C-cap present but no D/E sidechain closure site found for lariat.")

    rw = Chem.RWMol(mol)
    tail_res_idx = len(matches) - 1
    map_head = 9001
    map_tail = 9002
    map_closure = 9003

    if closure_mode == "sidechain_carbonyl":
        tail_res_idx = closure_res_idx if closure_res_idx is not None else tail_res_idx
        rw.GetAtomWithIdx(head_n).SetAtomMapNum(map_head)
        rw.GetAtomWithIdx(closure_atom).SetAtomMapNum(map_closure)
        head_idx = _find_atom_by_map(rw, map_head)
        clos_idx = _find_atom_by_map(rw, map_closure)
        _strip_h_from_n(rw, head_idx)
        rw.AddBond(head_idx, clos_idx, rdchem.BondType.SINGLE)
        _strip_oh_from_c(rw, clos_idx)
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
        rw.AddBond(tail_idx, clos_idx, rdchem.BondType.SINGLE)
        _strip_oh_from_c(rw, tail_idx)

    for atom in rw.GetAtoms():
        if atom.GetAtomMapNum() in (map_head, map_tail, map_closure):
            atom.SetAtomMapNum(0)

    # 需要 C-cap 时，在末端主链羧基碳上拼接（去掉占位符）
    if c_cap:
        # 删除/新增原子后索引可能变化，重新定位 backbone
        refreshed = list(get_backbone_atoms(rw.GetMol()))
        if not refreshed:
            raise ValueError("Backbone not found when attaching C-cap.")
        if tail_res_idx >= len(refreshed):
            tail_res_idx = len(refreshed) - 1
        c_idx = refreshed[tail_res_idx][2]
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

    _fix_overvalent_n(rw)
    final_mol = rw.GetMol()
    Chem.SanitizeMol(final_mol)
    return Chem.MolToSmiles(final_mol, isomericSmiles=True)


def main():
    parser = argparse.ArgumentParser(description="Lariat sequence -> SMILES (linear approx)")
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
