#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared utilities for peptide sequence conversion
"""

import rdkit
from rdkit import Chem
from rdkit.Chem import rdchem
from typing import Optional, Tuple, List, Set

def remove_atom_maps(smiles: str) -> str:
    """Remove atom mapping numbers from SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def clean_smiles(smiles: str) -> str:
    """Clean and standardize SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def get_backbone_atoms(mol: Chem.Mol) -> Tuple[Tuple[int, int, int], ...]:
    """
    Get ordered peptide backbone atoms (N-CA-C(=O)) excluding side-chain amides.

    Residues are ordered from N-terminus to C-terminus by following peptide bonds.
    Side-chain amide branches are reported and skipped. The molecule is annotated with
    properties `_n_cap_is_H` and `_c_cap_is_H` indicating whether termini are hydrogen capped.
    """
    pattern = Chem.MolFromSmarts("[N;$(NCC(=O))]-[C;$(C(N)C=O)]-[C;$(C=O)]")
    raw_matches: List[Tuple[int, int, int]] = list(mol.GetSubstructMatches(pattern))
    if len(raw_matches) <= 1:
        _annotate_terminal_caps(mol, raw_matches)
        return tuple(raw_matches)

    def bond_connects(prev: Tuple[int, int, int], curr: Tuple[int, int, int]) -> bool:
        bond = mol.GetBondBetweenAtoms(prev[2], curr[0])
        if not bond or bond.GetBondType() != rdchem.BondType.SINGLE:
            return False
        carbon = mol.GetAtomWithIdx(prev[2])
        return any(
            b.GetBondType() == rdchem.BondType.DOUBLE
            and b.GetOtherAtom(carbon).GetAtomicNum() == 8
            for b in carbon.GetBonds()
        )#确定是N-C=O

    # # Report discontinuities based on the initial ordering (sorted by atom index)
    # sorted_matches = [
    #     match for _, match in sorted(
    #         enumerate(raw_matches), key=lambda item: min(item[1])
    #     )
    # ]
    # for idx in range(1, len(sorted_matches)):
    #     prev = sorted_matches[idx - 1]
    #     curr = sorted_matches[idx]
    #     if not bond_connects(prev, curr):
    #         remaining = len(sorted_matches) - idx
    #         # print(
    #         #     f"[get_backbone_atoms] branch detected at residue index {idx} (1-based {idx + 1}); "
    #         #     f"remaining residues in branch: {remaining}"
    #         # )

    backbone_atoms: Set[int] = {atom for match in raw_matches for atom in match}
    match_by_n = {match[0]: match for match in raw_matches}

    def is_n_terminal(match: Tuple[int, int, int]) -> bool:
        n_idx = match[0]
        n_atom = mol.GetAtomWithIdx(n_idx)
        neighbors = [
            nb.GetIdx()
            for nb in n_atom.GetNeighbors()
            if nb.GetAtomicNum() > 1 and nb.GetIdx() in backbone_atoms
        ]
        return len(neighbors) == 1  # only the Cα within backbone, 除非是backbone cycle peptide all NeiN=2

    n_terminal_candidates = [m for m in raw_matches if is_n_terminal(m)]
    if not n_terminal_candidates:
        n_terminal_candidates = raw_matches

    def traverse(start: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        visited: Set[int] = set()
        ordered: List[Tuple[int, int, int]] = []
        current = start
        while current:
            key = current[0]
            if key in visited:
                break
            visited.add(key)
            ordered.append(current)
            c_idx = current[2]
            next_match = None
            for nb in mol.GetAtomWithIdx(c_idx).GetNeighbors():
                n_idx = nb.GetIdx()
                candidate = match_by_n.get(n_idx)
                if candidate and candidate[0] not in visited and bond_connects(current, candidate):
                    next_match = candidate
                    break
            current = next_match#go with setup bridge
        return ordered

    best_chain: List[Tuple[int, int, int]] = []
    for candidate in n_terminal_candidates:
        chain = traverse(candidate)
        if len(chain) >= len(best_chain):
            best_chain = chain

    if not best_chain:
        print(f"check traverse process")
        best_chain = raw_matches

    _annotate_terminal_caps(mol, best_chain)
    return tuple(best_chain)


def _annotate_terminal_caps(
    mol: Chem.Mol, matches: List[Tuple[int, int, int]]
) -> None:
    """Annotate molecule properties indicating whether termini are hydrogen capped."""
    if not matches:
        mol.SetProp("_n_cap_is_H", "true")
        mol.SetProp("_c_cap_is_H", "true")
        return

    backbone_atoms = {idx for match in matches for idx in match}

    # N-terminus
    n_idx = matches[0][0]
    n_atom = mol.GetAtomWithIdx(n_idx)
    n_heavy_neighbors = [
        nb.GetIdx()
        for nb in n_atom.GetNeighbors()
        if nb.GetAtomicNum() > 1 and nb.GetIdx() not in backbone_atoms
    ]
    mol.SetProp("_n_cap_is_H", "true" if not n_heavy_neighbors else "false")

    # C-terminus
    c_idx = matches[-1][2]
    c_atom = mol.GetAtomWithIdx(c_idx)
    c_heavy_neighbors = []
    for nb in c_atom.GetNeighbors():
        idx = nb.GetIdx()
        if idx == matches[-1][1]:
            continue  # skip alpha carbon
        if nb.GetAtomicNum() > 1:
            bond = mol.GetBondBetweenAtoms(c_idx, idx)
            if bond and bond.GetBondType() == rdchem.BondType.DOUBLE and nb.GetAtomicNum() == 8:
                continue  # ignore carbonyl oxygen
            c_heavy_neighbors.append(idx)
    mol.SetProp("_c_cap_is_H", "true" if not c_heavy_neighbors else "false")

def find_alpha_carbon(mol: Chem.Mol) -> Optional[int]:
    """Find alpha carbon atom index"""
    pattern = Chem.MolFromSmarts("[N;$(NC)]-[C;$(C(N)C=O)]-[C;$(C=O)]")
    matches = mol.GetSubstructMatches(pattern)
    if matches:
        return matches[0][1]  # CA is the middle atom
    return None

# Common caps and modifications
N_CAPS = {
    "ac": "CC(=O)N",      # Acetyl
    "formyl": "C(=O)N",   # Formyl
}

C_CAPS = {
    "am": "NC(=O)",       # Amide (-CONH2)
    "ome": "COC(=O)",     # Methyl ester
}

# Amino acid name mappings
AA1TO3 = {
    "A": "Ala", "C": "Cys", "D": "Asp", "E": "Glu", "F": "Phe",
    "G": "Gly", "H": "His", "I": "Ile", "K": "Lys", "L": "Leu",
    "M": "Met", "N": "Asn", "P": "Pro", "Q": "Gln", "R": "Arg",
    "S": "Ser", "T": "Thr", "V": "Val", "W": "Trp", "Y": "Tyr"
}

AA3TO1 = {v: k for k, v in AA1TO3.items()}
