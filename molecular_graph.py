"""Standardized parent-graph identity used to validate peptide reconstruction."""
import re
from typing import Optional
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def canonical_standardized(smiles: str) -> Optional[str]:
    s = str(smiles or "").strip()
    if not s:
        return None
    s = re.sub(r":\d+(?=\])", "", s)
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
    except Exception:
        pass
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
