#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fragment utilities built on top of smi2seq.

Provides a helper to split peptide SMILES into residue fragments while honouring
the monomer library (core + optional extend/custom). If a residue was matched
via fallback, the normalised fragment is returned.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from rdkit import Chem

from smi2seq import SMILES2Sequence

__all__ = ["smi2frags"]

_DEFAULT_LIB = Path("data/monomersFromHELMCoreLibrary.json")
_CONVERTER: Optional[SMILES2Sequence] = None


def _get_converter(lib_path: Optional[Union[str, Path]] = None) -> SMILES2Sequence:
    global _CONVERTER
    if lib_path:
        return SMILES2Sequence(lib_path=lib_path)
    if _CONVERTER is None:
        if _DEFAULT_LIB.exists():
            _CONVERTER = SMILES2Sequence(lib_path=_DEFAULT_LIB)
        else:
            _CONVERTER = SMILES2Sequence()
    return _CONVERTER


def _smiles_to_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse fragment SMILES: {smiles}")
    Chem.SanitizeMol(mol)
    return mol


def smi2frags(
    smiles: str,
    converter: Optional[SMILES2Sequence] = None,
    lib_path: Optional[Union[str, Path]] = None,
) -> List[Tuple[str, str, Chem.Mol]]:
    """
    Split a peptide SMILES string into monomer fragments registered in the library.

    Parameters
    ----------
    smiles : str
        Peptide SMILES string (caps and staples supported).

    Returns
    -------
    list of (code, fragment_smiles, fragment_mol)
        Code is the monomer symbol (e.g., "A", "ac", custom template name).
        Fragment SMILES is the HELM-style canonical string registered in the library.
        Fragment molecule is an RDKit Mol instantiated from the fragment SMILES.
    """
    smi_converter = converter or _get_converter(lib_path)
    _, details = smi_converter.convert(smiles, return_details=True)
    if not details:
        return []

    fragments: List[Tuple[str, str, Chem.Mol]] = []
    residues: Sequence[dict] = details.get("residues", [])  # type: ignore[arg-type]
    for residue in residues:
        code = residue.get("code", "X")
        frag_smiles = residue.get("canonical") or residue.get("canonical_smiles")
        if not frag_smiles:
            continue
        try:
            frag_mol = _smiles_to_mol(frag_smiles)
        except Exception:
            continue
        fragments.append((code, frag_smiles, frag_mol))
    return fragments
