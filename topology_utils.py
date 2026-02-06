#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for classifying peptide topologies."""

from __future__ import annotations

from typing import Optional

from rdkit import Chem

from utils import clean_smiles, get_backbone_atoms


def backbone_profile(smiles: str) -> Optional[dict]:
    """Return backbone connectivity statistics for a peptide SMILES."""
    cleaned = clean_smiles(smiles)
    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # pylint: disable=broad-except
        return None
    matches = list(get_backbone_atoms(mol))
    if not matches:
        return None
    backbone_atoms = {idx for trio in matches for idx in trio}
    n_counts = []
    c_counts = []
    for n_idx, _, c_idx in matches:
        n_atom = mol.GetAtomWithIdx(n_idx)
        c_atom = mol.GetAtomWithIdx(c_idx)
        n_counts.append(sum(1 for nb in n_atom.GetNeighbors() if nb.GetIdx() in backbone_atoms))
        c_counts.append(sum(1 for nb in c_atom.GetNeighbors() if nb.GetIdx() in backbone_atoms))
    n_singletons = [idx + 1 for idx, count in enumerate(n_counts) if count <= 1]
    c_singletons = [idx + 1 for idx, count in enumerate(c_counts) if count <= 1]
    head_to_tail = min(n_counts) >= 2 and min(c_counts) >= 2
    has_cycle = head_to_tail
    return {
        "n_counts": n_counts,
        "c_counts": c_counts,
        "residues": len(matches),
        "n_singletons": n_singletons,
        "c_singletons": c_singletons,
        "has_cycle": has_cycle,
        "head_to_tail": head_to_tail,
    }


def classify_topology(details: Optional[dict], profile: Optional[dict]) -> Optional[str]:
    """Infer topology label from converter details and backbone profile."""
    if isinstance(details, dict):
        cycle_meta = details.get("cycle_meta")
        if isinstance(cycle_meta, dict):
            annotation = cycle_meta.get("annotation")
            if annotation:
                return annotation
        if details.get("staple_pose"):
            return "stapled"
        if details.get("cyclized"):
            return "head2tail"
    if not profile:
        return None
    if profile.get("head_to_tail"):
        return "head2tail"
    if is_valid_linear_profile(profile):
        return "linear"
    return None


def is_valid_linear_profile(profile: Optional[dict]) -> bool:
    """Return True when the profile represents a single linear peptide chain."""
    if not profile:
        return False
    if profile.get("residues", 0) == 0:
        return False
    if len(profile.get("n_singletons", [])) != 1:
        return False
    if len(profile.get("c_singletons", [])) != 1:
        return False
    return True
