#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smi2seq_cycle
=============

Convert a cyclic peptide SMILES string back to a sequence representation.
The converter breaks a single peptide bond to linearise the molecule and
then reuses the SMILES2Sequence pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from rdkit import Chem

from smi2seq import SMILES2Sequence
from disulfide_utils import (
    annotate_sequence_with_disulfides,
    detect_disulfide_pairs,
    split_sequence_metadata,
    merge_sequence_metadata,
)
from utils import clean_smiles
from topology_utils import backbone_profile, classify_topology, is_valid_linear_profile

DEFAULT_LIB_PATH = Path("data/monomersFromHELMCoreLibrary.json")


def _init_converter(converter: SMILES2Sequence | None = None) -> SMILES2Sequence:
    if converter is not None:
        return converter
    if DEFAULT_LIB_PATH.exists():
        return SMILES2Sequence(DEFAULT_LIB_PATH)
    return SMILES2Sequence()


def _is_carbonyl_carbon(atom: Chem.Atom) -> bool:
    if atom.GetAtomicNum() != 6:
        return False
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetAtomicNum() == 8 and bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            return True
    return False


def _find_peptide_bonds(mol: Chem.Mol) -> List[Tuple[int, int]]:
    bonds: List[Tuple[int, int]] = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
            continue
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        if a.GetAtomicNum() == 6 and b.GetAtomicNum() == 7:
            if _is_carbonyl_carbon(a):
                bonds.append((a.GetIdx(), b.GetIdx()))
        elif a.GetAtomicNum() == 7 and b.GetAtomicNum() == 6:
            if _is_carbonyl_carbon(b):
                bonds.append((b.GetIdx(), a.GetIdx()))
    return bonds


def _rebuild_linear_candidate(mol: Chem.Mol, c_idx: int, n_idx: int) -> Optional[str]:
    rw = Chem.RWMol(mol)
    if rw.GetBondBetweenAtoms(c_idx, n_idx) is None:
        return None
    rw.RemoveBond(c_idx, n_idx)
    n_h_idx = rw.AddAtom(Chem.Atom(1))
    rw.AddBond(n_idx, n_h_idx, Chem.rdchem.BondType.SINGLE)
    o_idx = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(c_idx, o_idx, Chem.rdchem.BondType.SINGLE)
    o_h_idx = rw.AddAtom(Chem.Atom(1))
    rw.AddBond(o_idx, o_h_idx, Chem.rdchem.BondType.SINGLE)
    try:
        Chem.SanitizeMol(rw)
    except Exception:  # pylint: disable=broad-except
        return None
    return Chem.MolToSmiles(rw, isomericSmiles=True)


def _linearise_cycle(smiles: str) -> Tuple[str, dict]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    peptide_bonds = _find_peptide_bonds(mol)
    if len(peptide_bonds) < 2:
        raise ValueError("Unable to locate sufficient peptide bonds for a cyclic peptide.")

    candidates: List[Tuple[Tuple[int, int], str, dict]] = []
    for c_idx, n_idx in peptide_bonds:
        linear_smiles = _rebuild_linear_candidate(mol, c_idx, n_idx)
        if not linear_smiles:
            continue
        profile = backbone_profile(linear_smiles)
        if not is_valid_linear_profile(profile):
            continue
        candidates.append(((c_idx, n_idx), linear_smiles, profile))

    if not candidates:
        raise ValueError("Unable to linearise cyclic peptide; no valid peptide bond break found.")

    def _score(entry: Tuple[Tuple[int, int], str, dict]) -> Tuple[int, int]:
        _, _, profile = entry
        n_term = min(profile.get("n_singletons", [0]))
        c_term = max(profile.get("c_singletons", [0]))
        return (n_term, -c_term)

    break_info, linear, profile = min(candidates, key=_score)
    metadata = {"break_c_idx": int(break_info[0]), "break_n_idx": int(break_info[1]), "linear_profile": profile}
    return linear, metadata


def smi2seq_cycle(
    smiles: str,
    converter: SMILES2Sequence | None = None,
) -> Tuple[str, dict]:
    converter = _init_converter(converter)
    cleaned = clean_smiles(smiles)
    raw_profile = backbone_profile(cleaned)
    linear_smiles, meta = _linearise_cycle(cleaned)
    sequence, details = converter.convert(linear_smiles, return_details=True)
    residue_count = (
        len(details.get("residues", [])) if isinstance(details, dict) else 0
    )
    cys_indices, ss_pairs = detect_disulfide_pairs(linear_smiles, residue_count)
    sequence = annotate_sequence_with_disulfides(
        sequence,
        details.get("n_cap") if isinstance(details, dict) else None,
        details.get("c_cap") if isinstance(details, dict) else None,
        cys_indices,
        ss_pairs,
    )
    topology = classify_topology(details, raw_profile) or "head2tail"
    meta.update({"was_cyclic": True, "topology": topology})
    if isinstance(details, dict):
        merged = dict(details)
        merged["cycle_meta"] = meta
    else:
        merged = {"details": details, "cycle_meta": meta}
    core, metadata = split_sequence_metadata(sequence)
    metadata = [item for item in metadata if item.lower() not in {"head2tail", "lariat"}]
    metadata.append("lariat" if topology == "lariat" else "head2tail")
    annotated = merge_sequence_metadata(core, metadata)
    return annotated, merged


def _load_smiles(path: Path) -> List[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cyclic peptide SMILES → sequence converter")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("smi2seq_cycle_input.txt"),
        help="Input file with one SMILES per line.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("smi2seq_cycle_out.txt"),
        help="File to write sequences + metadata (tab separated).",
    )
    args = parser.parse_args()

    smiles_list = _load_smiles(args.input)
    converter = _init_converter()
    rows: List[str] = []
    for smi in smiles_list:
        seq, info = smi2seq_cycle(smi, converter)
        rows.append(f"{smi}\t{seq}\t{info}")
    args.output.write_text("\n".join(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
