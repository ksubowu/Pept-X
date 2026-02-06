#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight peptide conformer and secondary-structure helper.

The implementation purposefully favours robustness over physical accuracy:
  * sequences are converted to SMILES with the existing seq2smi pipeline;
  * RDKit ETKDG v3 is used to generate one 3D conformer (UFF relaxation);
  * backbone Cα positions are inspected to assign coarse secondary-structure
    labels (helix / beta / coil) using a dihedral-based heuristic.

The module exposes a single convenience function,
`predict_secondary_structure`, which accepts either a peptide sequence
or a SMILES string and returns:
  - the canonical sequence (when available),
  - the canonical SMILES used for embedding,
  - per-residue secondary-structure labels,
  - 3D coordinates for the embedded conformer.

The function is dependency-free beyond RDKit and therefore can be imported
by both the legacy scripts under `peptseq_smi/` and the repackaged
`PeptSeqSmi` module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

from utils import get_backbone_atoms

DEFAULT_LIB = Path(__file__).resolve().parent / "data" / "monomersFromHELMCoreLibrary.json"


@dataclass
class ConformerResult:
    sequence: Optional[str]
    smiles: str
    secondary_structure: List[str]
    coordinates: List[Tuple[str, float, float, float]]
    residue_map: List[int]


def _ensure_mol_with_conformer(mol: Chem.Mol) -> Chem.Mol:
    """Add hydrogens, embed a 3D conformer, and perform a quick relaxation."""
    mol = Chem.AddHs(mol, addCoords=True)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError("Failed to embed peptide conformer.")
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    return mol


def _sequence_to_smiles(sequence: str, lib_path: Optional[str] = None) -> str:
    # Lazy import to avoid circular dependency when seq2smi also imports conformer.
    from seq2smi import MonomerLib, seq2smi  # pylint: disable=import-outside-toplevel

    lib = MonomerLib(lib_path or str(DEFAULT_LIB))
    return seq2smi(sequence, lib)


def _extract_ca_positions(mol: Chem.Mol) -> Tuple[List[int], List[Tuple[float, float, float]]]:
    matches = list(get_backbone_atoms(mol))
    if not matches:
        return [], []
    conformer = mol.GetConformer()
    indices: List[int] = []
    coords: List[Tuple[float, float, float]] = []
    for idx, (_, ca_idx, _) in enumerate(matches, start=1):
        position = conformer.GetAtomPosition(ca_idx)
        indices.append(idx)
        coords.append((position.x, position.y, position.z))
    return indices, coords


def _vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vec_norm(a):
    return math.sqrt(_vec_dot(a, a))


def _vec_scale(a, factor):
    return (a[0] * factor, a[1] * factor, a[2] * factor)


def _vec_normalize(a):
    norm = _vec_norm(a)
    if norm == 0:
        return (0.0, 0.0, 0.0)
    return _vec_scale(a, 1.0 / norm)


def _dihedral(p0, p1, p2, p3) -> float:
    """Return torsion angle in degrees for four XYZ tuples."""
    b0 = _vec_sub(p1, p0)
    b1 = _vec_sub(p2, p1)
    b2 = _vec_sub(p3, p2)

    b1 = _vec_normalize(b1)
    v = _vec_sub(b0, _vec_scale(b1, _vec_dot(b0, b1)))
    w = _vec_sub(b2, _vec_scale(b1, _vec_dot(b2, b1)))
    x = _vec_dot(v, w)
    y = _vec_dot(_vec_cross(b1, v), w)
    angle = math.degrees(math.atan2(y, x))
    return angle


def _assign_secondary_structure(coords: Sequence[Tuple[float, float, float]]) -> List[str]:
    """Classify residues using a torsion-based heuristic."""
    n = len(coords)
    if n == 0:
        return []
    if n < 4:
        return ["C"] * n

    labels = ["C"] * n
    torsions = []
    for i in range(n - 3):
        torsions.append(_dihedral(coords[i], coords[i + 1], coords[i + 2], coords[i + 3]))

    for i, tau in enumerate(torsions, start=1):
        if -90 <= tau <= -30:
            labels[i] = "H"
        elif tau <= -120 or tau >= 120:
            labels[i] = "E"
        else:
            labels[i] = "C"

    # extend classification to adjacent residues for smoother appearance
    for i in range(1, n - 1):
        if labels[i] == "H" and labels[i - 1] == "C":
            labels[i - 1] = "H"
        if labels[i] == "E" and labels[i - 1] == "C":
            labels[i - 1] = "E"

    return labels


def predict_secondary_structure(
    sequence: Optional[str] = None,
    smiles: Optional[str] = None,
    lib_path: Optional[str] = None,
) -> ConformerResult:
    """
    Generate a peptide conformer and infer secondary-structure labels.

    Parameters
    ----------
    sequence : str, optional
        Peptide sequence recognised by seq2smi (caps, staples allowed).
    smiles : str, optional
        Peptide SMILES. Takes precedence over `sequence` when provided.
    lib_path : str, optional
        Custom monomer library for seq2smi conversion.

    Returns
    -------
    ConformerResult
        Dataclass containing canonical sequence (if recovered),
        canonical SMILES, per-residue secondary-structure labels, atom
        coordinates, and a mapping of CA indices.
    """
    if not sequence and not smiles:
        raise ValueError("Either sequence or smiles must be provided.")

    resolved_sequence: Optional[str] = None
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Failed to parse SMILES for structure prediction.")
        smiles_for_embedding = Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        smiles_for_embedding = _sequence_to_smiles(sequence, lib_path=lib_path)
        resolved_sequence = sequence
        mol = Chem.MolFromSmiles(smiles_for_embedding)
        if mol is None:
            raise ValueError("Failed to build molecule from sequence.")

    mol = _ensure_mol_with_conformer(mol)
    ca_indices, ca_coords = _extract_ca_positions(mol)
    secondary = _assign_secondary_structure(ca_coords)

    conformer = mol.GetConformer()
    atom_coords = []
    for atom in mol.GetAtoms():
        pos = conformer.GetAtomPosition(atom.GetIdx())
        atom_coords.append((atom.GetSymbol(), pos.x, pos.y, pos.z))

    return ConformerResult(
        sequence=resolved_sequence,
        smiles=smiles_for_embedding,
        secondary_structure=secondary,
        coordinates=atom_coords,
        residue_map=ca_indices,
    )


def serialize_prediction(result: ConformerResult) -> dict:
    """Convert ConformerResult into a JSON-serialisable dictionary."""
    return {
        "sequence": result.sequence,
        "smiles": result.smiles,
        "secondary_structure": result.secondary_structure,
        "coordinates": result.coordinates,
        "residue_map": result.residue_map,
    }
