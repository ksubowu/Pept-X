#!/usr/bin/env python3
"""SMILES → lariat peptide sequence (standalone converter).

核心思路与 `lariat_test.py` 保持一致：
- 识别 backbone N-CA-C 序列；
- 找到 iso-peptide 键（侧链 C(=O)–N）以及与 Ccap 相连的键；
- 将肽键 + iso-peptide + Ccap 键全部切断，得到独立残基与帽基片段；
- 对片段进行“补 OH / 去 dummy / 统一规范化”后与单体库精确匹配；
- 输出序列，若存在 Ccap 则附加 `[C-cap:<smiles>]`，末尾加带残基位置的
  ``|lariat_n:n1-n2`` 标签；
- 未匹配的残基会记录并输出便于调试。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdchem import BondType

from seq2smi import MonomerLib  # 单体库
from utils import clean_smiles, get_backbone_atoms


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LIB = BASE_DIR / "data" / "monomersFromHELMCoreLibrary.json"
UNMATCHED_LOG = BASE_DIR / "smi2seq_lariat_unmatched.txt"


def _format_lariat_tag(kind: int, first: int, second: int) -> str:
    """Return the canonical explicit lariat tag."""

    if kind not in (1, 2, 3):
        raise ValueError(f"Unsupported lariat type: {kind}")
    return f"lariat_{kind}:{first}-{second}"


# ------------------------ 核心工具函数 ------------------------ #
def _canonical_mol(mol: Chem.Mol) -> Optional[str]:
    """统一的规范化：去 atom map、去显式 H、Sanitize，返回 canonical SMILES。"""
    if mol is None:
        return None
    m = Chem.Mol(mol)
    for atom in m.GetAtoms():
        atom.SetAtomMapNum(0)
    m = Chem.RemoveHs(m)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)


def _canonical(smiles: str) -> Optional[str]:
    if not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    return _canonical_mol(m)


def _build_canon_to_code(lib: MonomerLib) -> Dict[str, str]:
    canon_to_code: Dict[str, str] = {}
    for code, entry in lib.by_code.items():
        smi = entry.get("smiles") or entry.get("smiles_L") or entry.get("smiles_D")
        if isinstance(smi, list):
            smi = next((s for s in smi if isinstance(s, str)), None)
        canon = _canonical(smi)
        if canon:
            canon_to_code[canon] = entry.get("code", code)
    return canon_to_code


def _find_iso_peptide(
    mol: Chem.Mol,
    backbone: Sequence[Tuple[int, int, int]],
    atom_to_residue: Dict[int, int],
):
    """寻找 iso-peptide 键：backbone N 与非-backbone 的 C(=O) 且该 C 属于某个残基侧链。"""
    bb_atoms = {idx for triplet in backbone for idx in triplet}
    iso = {}
    for res_idx, triplet in enumerate(backbone):
        n_idx = triplet[0]
        atom_n = mol.GetAtomWithIdx(n_idx)
        for nb in atom_n.GetNeighbors():
            if nb.GetAtomicNum() == 6 and nb.GetIdx() not in bb_atoms:
                # 确认是 C=O
                has_double_o = any(
                    b.GetBondType() == BondType.DOUBLE and b.GetOtherAtom(nb).GetAtomicNum() == 8
                    for b in nb.GetBonds()
                )
                if has_double_o:
                    # 仅当该 C 属于某个残基侧链时视为异肽键（排除 N-cap）
                    if nb.GetIdx() in atom_to_residue:
                        iso[res_idx] = [n_idx, nb.GetIdx()]
    return iso


def _find_c_cap_bonds(
    mol: Chem.Mol,
    backbone: Sequence[Tuple[int, int, int]],
    atom_to_residue: Dict[int, int],
    ester_bonds: Optional[Dict[int, List[int]]] = None,
):
    """寻找 Ccap：末端或内段的 C 与非-backbone 单键相连。"""
    bb_atoms = {idx for triplet in backbone for idx in triplet}
    ccap = {}
    for res_idx, triplet in enumerate(backbone):
        c_idx = triplet[-1]
        atom_c = mol.GetAtomWithIdx(c_idx)
        for bond in atom_c.GetBonds():
            if bond.GetBondType() == BondType.DOUBLE:
                continue
            other = bond.GetOtherAtom(atom_c)
            # A terminal carboxyl OH/O- belongs to the free backbone terminus,
            # not to a separate C-cap monomer.
            if other.GetAtomicNum() == 8 and other.GetDegree() == 1:
                continue
            if other.GetIdx() not in bb_atoms:
                if ester_bonds and res_idx in ester_bonds and other.GetIdx() in ester_bonds[res_idx]:
                    continue
                if other.GetIdx() in atom_to_residue:
                    continue
                ccap[res_idx] = [c_idx, other.GetIdx()]
    return ccap


def _find_n_cap_bonds(
    mol: Chem.Mol,
    backbone: Sequence[Tuple[int, int, int]],
    atom_to_residue: Dict[int, int],
):
    """寻找 Ncap：backbone N 与非-backbone 的酰基碳相连。"""
    bb_atoms = {idx for triplet in backbone for idx in triplet}
    ncap = {}
    for res_idx, triplet in enumerate(backbone):
        n_idx = triplet[0]
        atom_n = mol.GetAtomWithIdx(n_idx)
        for bond in atom_n.GetBonds():
            if bond.GetBondType() == BondType.DOUBLE:
                continue
            other = bond.GetOtherAtom(atom_n)
            if other.GetAtomicNum() != 6:
                continue
            if other.GetIdx() in bb_atoms or other.GetIdx() in atom_to_residue:
                continue
            # require carbonyl C
            has_double_o = any(
                b.GetBondType() == BondType.DOUBLE and b.GetOtherAtom(other).GetAtomicNum() == 8
                for b in other.GetBonds()
            )
            if has_double_o:
                ncap[res_idx] = [n_idx, other.GetIdx()]
    return ncap


def _build_atom_to_residue(mol: Chem.Mol, backbone: Sequence[Tuple[int, int, int]]) -> Dict[int, int]:
    """Map side-chain atoms to residue index (exclude backbone traversal)."""
    bb_atoms = {idx for triplet in backbone for idx in triplet}
    atom_to_res: Dict[int, int] = {}
    for res_idx, (_, ca_idx, _) in enumerate(backbone):
        stack = [ca_idx]
        visited = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            atom = mol.GetAtomWithIdx(cur)
            for nb in atom.GetNeighbors():
                nb_idx = nb.GetIdx()
                if nb_idx in visited:
                    continue
                if nb_idx in bb_atoms:
                    continue
                atom_to_res[nb_idx] = res_idx
                stack.append(nb_idx)
    return atom_to_res


def _order_residues(mol: Chem.Mol, backbone: Sequence[Tuple[int, int, int]]):
    prev_map: Dict[int, int] = {}
    next_map: Dict[int, int] = {}
    for idx, res in enumerate(backbone):
        for jdx, other in enumerate(backbone):
            if idx == jdx:
                continue
            if mol.GetBondBetweenAtoms(other[0], res[-1]):
                next_map[idx] = jdx
            if mol.GetBondBetweenAtoms(other[-1], res[0]):
                prev_map[idx] = jdx

    start_idx = None
    for idx in range(len(backbone)):
        if idx not in prev_map:
            start_idx = idx
            break
    if start_idx is None:
        return sorted(range(len(backbone)), key=lambda i: backbone[i][0])
    order = [start_idx]
    seen = {start_idx}
    while order[-1] in next_map:
        nxt = next_map[order[-1]]
        if nxt in seen:
            break
        order.append(nxt)
        seen.add(nxt)
    if len(order) != len(backbone):
        return sorted(range(len(backbone)), key=lambda i: backbone[i][0])
    return order


def _cap_fragment(fragment: Chem.Mol) -> Chem.Mol:
    """Restore only groups at cut bonds; retain all uncut side-chain chemistry.

    A carbonyl at a cut amide/ester bond becomes COOH. An uncut aldehyde or
    ketone stays unchanged; charged oxygens (e.g. nitro O-) are never protonated.
    """
    rw = Chem.RWMol(fragment)
    boundaries = set()
    for atom in rw.GetAtoms():
        atom.SetIntProp("_fragment_original_index", atom.GetIdx())
        if atom.GetAtomicNum() == 0:
            boundaries.update(a.GetIdx() for a in atom.GetNeighbors())
    for idx in sorted((a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0), reverse=True):
        rw.RemoveAtom(idx)
    for atom in list(rw.GetAtoms()):
        if atom.GetIntProp("_fragment_original_index") not in boundaries:
            continue
        if atom.GetAtomicNum() == 6:
            carbonyl = any(b.GetBondType() == BondType.DOUBLE and b.GetOtherAtom(atom).GetAtomicNum() == 8
                           for b in atom.GetBonds())
            if carbonyl:
                oxygen = Chem.Atom(8)
                oxygen.SetNumExplicitHs(1)
                oxygen.SetNoImplicit(True)
                idx = rw.AddAtom(oxygen)
                rw.AddBond(atom.GetIdx(), idx, BondType.SINGLE)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
        atom.SetNumRadicalElectrons(0)
    capped = rw.GetMol()
    Chem.SanitizeMol(capped)
    return capped


def _match_fragment(frag: Chem.Mol, canon_to_code: Dict[str, str]):
    capped = _cap_fragment(frag)
    canon = _canonical_mol(capped)
    code = canon_to_code.get(canon)
    if code == "Pro-al":
        # Prefer Pro unless we explicitly want Pro-al (avoid false positives).
        code = "P"
    return (code if code else "X", canon)


def _find_cap_fragment(atom_frags, frag_mols, cap_bonds, cap_like: bool = False):
    cap_smiles = None
    if cap_bonds:
        # 只取一个 Ccap
        _, (_, cap_idx) = next(iter(cap_bonds.items()))
        frag_idx = None
        for i, atom_ids in enumerate(atom_frags):
            if cap_idx in atom_ids:
                frag_idx = i
                break
        if frag_idx is not None:
            cap_mol = Chem.Mol(frag_mols[frag_idx])
            if cap_like:
                cap_smiles = _canonical_mol(_cap_fragment(cap_mol))
            else:
                rw = Chem.RWMol(cap_mol)
                dummies = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
                for d_idx in sorted(dummies, reverse=True):
                    if d_idx >= rw.GetNumAtoms():
                        continue
                    neighs = list(rw.GetAtomWithIdx(d_idx).GetNeighbors())
                    rw.RemoveAtom(d_idx)
                    for n in neighs:
                        if n.GetAtomicNum() == 7 and len(n.GetNeighbors()) < 3:
                            n.SetNoImplicit(False)
                            n.SetNumExplicitHs(1)
                cap_mol = rw.GetMol()
                try:
                    Chem.SanitizeMol(cap_mol)
                except Exception:
                    pass
                cap_smiles = _canonical_mol(cap_mol)
    return cap_smiles


def _normalize_c_cap_for_code(cap_mol: Chem.Mol) -> Optional[str]:
    """Normalize C-cap fragment for code lookup (strip terminal O attached to carbonyl C)."""
    if cap_mol is None:
        return None
    rw = Chem.RWMol(cap_mol)
    # Remove terminal O (degree 1) attached to carbonyl carbon (C=O)
    to_remove = []
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() != 8:
            continue
        if len(atom.GetNeighbors()) != 1:
            continue
        nb = atom.GetNeighbors()[0]
        if nb.GetAtomicNum() != 6:
            continue
        has_double_o = any(
            b.GetBondType() == BondType.DOUBLE and b.GetOtherAtom(nb).GetAtomicNum() == 8
            for b in nb.GetBonds()
        )
        if has_double_o:
            to_remove.append(atom.GetIdx())
            continue
        # Also strip sp3-like terminal O attached to ring carbon (e.g., O[C]1CCCCN1 from -pip)
        if not has_double_o and nb.GetIsAromatic() is False:
            to_remove.append(atom.GetIdx())
    for idx in sorted(to_remove, reverse=True):
        if idx < rw.GetNumAtoms():
            rw.RemoveAtom(idx)
    mol2 = rw.GetMol()
    try:
        Chem.SanitizeMol(mol2)
    except Exception:
        pass
    canon = _canonical_mol(mol2)
    if canon and "[C]" in canon:
        try:
            canon = _canonical_mol(Chem.MolFromSmiles(canon.replace("[C]", "C")))
        except Exception:
            pass
    return canon


def _template_backbone(mol: Chem.Mol, library) -> List[Tuple[int, int, int]]:
    """Infer backbone endpoints from exact stereochemical template subgraphs.

    This path handles beta backbones with amidated side chains and mixed
    peptide/depsipeptide backbones. Unmatched graph boundaries are permitted
    only at the registered attachment anchors; HELM/sequence references are
    never supplied to inference.
    """
    from seq2smi_v2 import build_fragment

    candidates = {}
    for entry in library.by_code.values():
        if str(entry.get("role", "")).lower() in {"ncap", "ccap"}:
            continue
        try:
            fragment, anchors, _, _ = build_fragment(entry, False)
        except (ValueError, RuntimeError, KeyError, TypeError):
            continue
        if not {1, 2} <= set(anchors):
            continue
        removed = {a.GetIdx() for a in fragment.GetAtoms()
                   if a.GetAtomMapNum() > 0 or a.GetAtomicNum() == 1}
        if any(idx in removed for idx in anchors.values()):
            continue
        remaining = [i for i in range(fragment.GetNumAtoms()) if i not in removed]
        index = {old: new for new, old in enumerate(remaining)}
        rw = Chem.RWMol(fragment)
        for idx in sorted(removed, reverse=True):
            rw.RemoveAtom(idx)
        query = rw.GetMol()
        query.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(query)
        n, c = index[anchors[1]], index[anchors[2]]
        path = Chem.GetShortestPath(query, n, c)
        if len(path) < 3:
            continue
        attachment_indices = {index[i] for i in anchors.values()}
        matches = mol.GetSubstructMatches(query, useChirality=True, maxMatches=10000)
        if len(matches) >= 10000:
            raise ValueError("Template backbone matching reached the search limit.")
        for match in matches:
            atoms = set(match)
            if any(any(nb.GetIdx() not in atoms for nb in mol.GetAtomWithIdx(target).GetNeighbors())
                   for qidx, target in enumerate(match) if qidx not in attachment_indices):
                continue
            triple = (match[n], match[path[1]], match[c])
            candidates[triple] = max(candidates.get(triple, 0), len(match))
    if not candidates:
        raise ValueError("No exact anchored monomer backbone matches.")
    triples = sorted(candidates)
    next_nodes = {bb: [other for other in triples if other != bb
                      and mol.GetBondBetweenAtoms(bb[2], other[0]) is not None]
                  for bb in triples}
    best = []
    best_score = (0, 0)
    visits = 0

    def visit(chain, used_n, used_c, coverage):
        nonlocal best, best_score, visits
        visits += 1
        if visits > 100000:
            raise ValueError("Ambiguous backbone traversal exceeded the search limit.")
        score = (len(chain), coverage)
        if score > best_score:
            best, best_score = list(chain), score
        for other in next_nodes[chain[-1]]:
            if other[0] not in used_n and other[2] not in used_c:
                visit(chain + [other], used_n | {other[0]}, used_c | {other[2]},
                      coverage + candidates[other])
    for bb in triples:
        visit([bb], {bb[0]}, {bb[2]}, candidates[bb])
    return best


def convert(
    smiles: str,
    lib: Optional[MonomerLib] = None,
    lib_path: Path = DEFAULT_LIB,
    list_output: bool = False,
) -> Tuple[str, dict]:
    """Recover a sequence from SMILES alone, with anchored-template inference
    for backbones the primary peptide traversal cannot resolve.
    """
    library = lib if lib is not None else MonomerLib(str(lib_path))
    initial = None
    error = None
    try:
        initial = _convert_with_backbone(smiles, library, lib_path, list_output)
        if not initial[1].get("unmatched") and initial[1].get("lariat_type"):
            initial[1]["backbone_inference"] = "peptide_traversal"
            return initial
    except (ValueError, RuntimeError, KeyError) as exc:
        error = exc
    try:
        # Match atom indices in exactly the same canonicalized molecule used
        # by the fragment converter below.
        mol = Chem.MolFromSmiles(clean_smiles(smiles))
        if mol is None:
            raise ValueError("Invalid SMILES")
        backbone = _template_backbone(mol, library)
        sequence, details = _convert_with_backbone(smiles, library, lib_path, list_output, backbone)
        details["backbone_inference"] = "template_anchors"
        if not details.get("unmatched") and details.get("lariat_type"):
            return sequence, details
    except (ValueError, RuntimeError, KeyError) as exc:
        if initial is None:
            raise
        initial[1]["template_backbone_error"] = f"{type(exc).__name__}: {exc}"
    if initial is not None:
        initial[1]["backbone_inference"] = "peptide_traversal"
        unmatched = initial[1].get("unmatched", [])
        if unmatched:
            lines = [f"{smiles}\tresidue={u['residue_index']}\tatoms={u['atom_indices']}\tsmiles={u['smiles']}"
                     for u in unmatched]
            with UNMATCHED_LOG.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
        return initial
    raise error


# ------------------------ 转换主流程 ------------------------ #
def _convert_with_backbone(
    smiles: str,
    lib: Optional[MonomerLib] = None,
    lib_path: Path = DEFAULT_LIB,
    list_output: bool = False,
    backbone_override=None,
) -> Tuple[str, dict]:
    cleaned = clean_smiles(smiles)
    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        raise ValueError("Invalid SMILES")

    backbone = list(backbone_override) if backbone_override is not None else list(get_backbone_atoms(mol))
    if not backbone:
        raise ValueError("No peptide backbone detected")

    atom_to_residue = _build_atom_to_residue(mol, backbone)
    iso_peptide = _find_iso_peptide(mol, backbone, atom_to_residue)
    # lactam lariat: backbone C single-bond to sidechain N of another residue (K/R)
    lactam_bonds: Dict[int, List[int]] = {}
    lactam_head = None
    lactam_tail = None
    for res_idx, triplet in enumerate(backbone):
        c_idx = triplet[-1]
        atom_c = mol.GetAtomWithIdx(c_idx)
        for bond in atom_c.GetBonds():
            if bond.GetBondType() == BondType.DOUBLE:
                continue
            other = bond.GetOtherAtom(atom_c)
            if other.GetAtomicNum() == 7 and other.GetIdx() in atom_to_residue and atom_to_residue[other.GetIdx()] != res_idx:
                lactam_bonds[res_idx] = [c_idx, other.GetIdx()]
                lactam_tail = res_idx
                lactam_head = atom_to_residue[other.GetIdx()]
                break
        if lactam_head is not None:
            break
    # ester lariat: backbone C single-bond to sidechain O of another residue
    ester_bonds: Dict[int, List[int]] = {}
    for res_idx, triplet in enumerate(backbone):
        c_idx = triplet[-1]
        atom_c = mol.GetAtomWithIdx(c_idx)
        for bond in atom_c.GetBonds():
            if bond.GetBondType() == BondType.DOUBLE:
                continue
            other = bond.GetOtherAtom(atom_c)
            if other.GetAtomicNum() == 8 and other.GetIdx() in atom_to_residue and atom_to_residue[other.GetIdx()] != res_idx:
                ester_bonds[res_idx] = [c_idx, other.GetIdx()]
    ncap_bonds = _find_n_cap_bonds(mol, backbone, atom_to_residue)
    ccap_bonds = _find_c_cap_bonds(mol, backbone, atom_to_residue, ester_bonds)

    # 肽键 + iso-peptide + Ccap 键索引
    bond_indices: List[int] = []
    for addbond in [iso_peptide, lactam_bonds, ester_bonds, ccap_bonds, ncap_bonds]:
        for _, v in addbond.items():
            a1, a2 = v[0], v[1]
            b_ = mol.GetBondBetweenAtoms(a1, a2)
            if b_:
                bond_indices.append(b_.GetIdx())
    # 正常肽键
    for idx in range(len(backbone) - 1):
        bond = mol.GetBondBetweenAtoms(backbone[idx][2], backbone[idx + 1][0])
        if bond:
            bond_indices.append(bond.GetIdx())

    fragmol = (
        rdmolops.FragmentOnBonds(mol, bond_indices, addDummies=True)
        if bond_indices
        else Chem.Mol(mol)
    )

    atom_frags = Chem.GetMolFrags(fragmol, asMols=False, sanitizeFrags=False)
    residues = [{"N": n, "CA": ca, "C": c} for (n, ca, c) in backbone]
    ca_to_fragment: Dict[int, int] = {}
    for frag_idx, atom_ids in enumerate(atom_frags):
        for residue in residues:
            if residue["CA"] in atom_ids:
                ca_to_fragment[residue["CA"]] = frag_idx
    if len(ca_to_fragment) != len(residues):
        raise ValueError("Failed to map fragments to residues.")

    frag_mols = list(Chem.GetMolFrags(fragmol, asMols=True, sanitizeFrags=True))

    library = lib if lib is not None else MonomerLib(str(lib_path))
    canon_to_code = _build_canon_to_code(library)

    # 序列顺序
    order = _order_residues(mol, backbone)
    bb_atoms = {idx for triplet in backbone for idx in triplet}
    n_methylated: Dict[int, bool] = {}
    for res_idx, triplet in enumerate(backbone):
        n_idx = triplet[0]
        atom_n = mol.GetAtomWithIdx(n_idx)
        has_extra_c = False
        for nb in atom_n.GetNeighbors():
            if nb.GetAtomicNum() != 6:
                continue
            if nb.GetIdx() in bb_atoms:
                continue
            bond = mol.GetBondBetweenAtoms(n_idx, nb.GetIdx())
            if bond is None or bond.GetBondType() != BondType.SINGLE:
                continue
            # avoid carbonyl or aromatic attachments (peptide bonds/caps)
            is_carbonyl = any(
                b.GetBondType() == BondType.DOUBLE and b.GetOtherAtom(nb).GetAtomicNum() == 8
                for b in nb.GetBonds()
            )
            if is_carbonyl or nb.GetIsAromatic():
                continue
            # require methyl-like carbon: only one heavy-atom neighbor (the N)
            heavy_neighbors = [x for x in nb.GetNeighbors() if x.GetAtomicNum() > 1]
            if len(heavy_neighbors) != 1:
                continue
            has_extra_c = True
            break
        n_methylated[res_idx] = has_extra_c

    tokens: List[str] = []
    unmatched: List[dict] = []
    residue_records: List[dict] = []
    # Lariat positions refer to AA/NCAA only; cap tokens never add an offset.
    output_position_by_residue = {
        residue_idx: output_position
        for output_position, residue_idx in enumerate(order, start=1)
    }
    for idx in order:
        ca = residues[idx]["CA"]
        frag_idx = ca_to_fragment[ca]
        code, canon = _match_fragment(frag_mols[frag_idx], canon_to_code)
        if n_methylated.get(idx):
            n_methyl_map = {
                "L": "meL",
                "dL": "Me_dL",
                "A": "meA",
                "dA": "Me_dA",
                "V": "meV",
                "dV": "Me_dV",
                "I": "meI",
                "dI": "Me_dI",
                "F": "meF",
                "dF": "Me_dF",
                "T": "meT",
                "dT": "Me_dT",
                "P": "meP",
                "dP": "Me_dP",
            }
            candidate = n_methyl_map.get(code)
            if candidate:
                resolved = library.resolve_code(candidate) or candidate
                if resolved.lower() in library.by_code:
                    code = resolved
        tokens.append(code)
        output_position = len(tokens)
        residue_records.append(
            {
                "index": output_position,
                "source_backbone_index": idx + 1,
                "code": code,
                "canonical": canon,
                "canonical_smiles": canon,
                "used_fallback": code == "X",
            }
        )
        if code == "X":
            unmatched.append({
                "residue_index": output_position,
                "source_backbone_index": idx + 1,
                "smiles": canon,
                "atom_indices": sorted(atom_frags[frag_idx]),
            })

    cap_smiles = _find_cap_fragment(atom_frags, frag_mols, ccap_bonds, cap_like=True)
    cap_code = canon_to_code.get(_canonical(cap_smiles)) if cap_smiles else None
    if cap_smiles and not cap_code:
        cap_alt = _normalize_c_cap_for_code(Chem.MolFromSmiles(cap_smiles))
        if cap_alt:
            cap_code = canon_to_code.get(cap_alt)
    ncap_smiles = _find_cap_fragment(atom_frags, frag_mols, ncap_bonds, cap_like=True)
    ncap_code = canon_to_code.get(_canonical(ncap_smiles)) if ncap_smiles else None

    seq_core = ".".join(tokens)
    ncap_token = None
    if ncap_smiles:
        if ncap_code:
            ncap_token = "ac-" if ncap_code == "ac" else ncap_code
            seq_core = f"{ncap_token}.{seq_core}"
        else:
            ncap_token = f"[N-cap:{ncap_smiles}]"
            seq_core = f"{ncap_token}.{seq_core}"
    cap_token = None
    if cap_smiles:
        if cap_code:
            cap_token = cap_code
            seq_core += f".{cap_token}"
        else:
            cap_token = f"[C-cap:{cap_smiles}]"
            seq_core += f".{cap_token}"
    lariat_tag = "lariat"
    lariat_kind = None
    lariat_positions = None
    lariat_atoms = None
    if iso_peptide:
        backbone_n_residue, lariat_atoms = next(iter(iso_peptide.items()))
        sidechain_residue = atom_to_residue.get(lariat_atoms[1])
        if sidechain_residue is not None:
            lariat_kind = 1
            lariat_positions = [
                output_position_by_residue[backbone_n_residue],
                output_position_by_residue[sidechain_residue],
            ]
    elif lactam_bonds:
        backbone_c_residue, lariat_atoms = next(iter(lactam_bonds.items()))
        sidechain_residue = atom_to_residue.get(lariat_atoms[1])
        if sidechain_residue is not None:
            lariat_kind = 2
            lariat_positions = [
                output_position_by_residue[sidechain_residue],
                output_position_by_residue[backbone_c_residue],
            ]
    elif ester_bonds:
        backbone_c_residue, lariat_atoms = next(iter(ester_bonds.items()))
        sidechain_residue = atom_to_residue.get(lariat_atoms[1])
        if sidechain_residue is not None:
            lariat_kind = 3
            lariat_positions = [
                output_position_by_residue[sidechain_residue],
                output_position_by_residue[backbone_c_residue],
            ]
    if lariat_kind and lariat_positions:
        lariat_tag = _format_lariat_tag(lariat_kind, *lariat_positions)

    if list_output:
        tokens_out: List[str] = []
        if ncap_token:
            tokens_out.append(ncap_token)
        tokens_out.extend(tokens)
        if cap_token:
            tokens_out.append(cap_token)
        sequence = f"{tokens_out}|{lariat_tag}"
    else:
        sequence = seq_core + f"|{lariat_tag}"

    details = {"unmatched": unmatched, "residues": residue_records} if unmatched else {"residues": residue_records}
    if cap_smiles:
        details["c_cap"] = cap_smiles
        if cap_code:
            details["c_cap_code"] = cap_code
    if iso_peptide:
        details["iso_peptide_bonds"] = iso_peptide
    if lactam_bonds:
        details["lactam_bonds"] = lactam_bonds
    if ester_bonds:
        details["ester_bonds"] = ester_bonds
    if lariat_kind and lariat_positions:
        details.update(
            {
                "lariat_type": f"lariat_{lariat_kind}",
                "lariat_positions": lariat_positions,
                "lariat_tag": lariat_tag,
                "lariat_atom_indices": lariat_atoms,
            }
        )

    return sequence, details


def main():
    parser = argparse.ArgumentParser(description="SMILES → lariat peptide sequence")
    parser.add_argument("--input", type=Path, default=Path("smi2seq_lariat_input.txt"))
    parser.add_argument("--output", type=Path, default=Path("smi2seq_lariat_out.txt"))
    parser.add_argument("--lib", type=Path, default=DEFAULT_LIB)
    parser.add_argument("--list-output", action="store_true", help="Emit Python list format sequence.")
    args = parser.parse_args()

    rows: List[str] = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        smi = line.strip()
        if not smi:
            continue
        try:
            seq, info = convert(smi, lib_path=args.lib, list_output=args.list_output)
            rows.append(f"{smi}\t{seq}\t{info}")
        except Exception as exc:  # pylint: disable=broad-except
            rows.append(f"{smi}\tERROR: {exc}")
    args.output.write_text("\n".join(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
