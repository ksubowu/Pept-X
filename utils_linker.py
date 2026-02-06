import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdFMCS

DEFAULT_LINKER_DICT = Path("data/linker_dict.json")


def normalize_linker_key(smiles: str) -> str:
    stripped = re.sub(r":\\d+", "", smiles)
    mol = Chem.MolFromSmiles(stripped)
    if mol is None:
        return stripped
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def load_linker_dict(path: Optional[Path] = None) -> Dict[str, object]:
    linker_path = path or DEFAULT_LINKER_DICT
    if not linker_path.exists():
        return {"version": 1, "entries": []}
    with linker_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def upsert_linker_entry(linker_smiles: str, path: Optional[Path] = None) -> bool:
    """Ensure linker_smiles exists in linker_dict.json; return True if added."""
    if not linker_smiles:
        return False
    linker_path = path or DEFAULT_LINKER_DICT
    data = load_linker_dict(linker_path)
    entries = data.setdefault("entries", [])
    hint_key = normalize_linker_key(linker_smiles)
    for entry in entries:
        keys = entry.get("keys", [])
        for key in keys:
            if normalize_linker_key(key) == hint_key:
                return False
        mapped = entry.get("mapped")
        if mapped and normalize_linker_key(mapped) == hint_key:
            return False
    next_id = len(entries) + 1
    entries.append(
        {
            "name": f"user_linker_{next_id}",
            "mapped": linker_smiles,
            "keys": [linker_smiles, hint_key],
        }
    )
    with linker_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return True


def build_linker_queries(linker_dict: Dict[str, object]) -> List[Tuple[Optional[Chem.Mol], str]]:
    entries = linker_dict.get("entries", [])
    queries: List[Tuple[Optional[Chem.Mol], str]] = []
    for entry in entries:
        pattern = entry.get("pattern_smarts")
        mapped = entry.get("mapped")
        if not mapped:
            continue
        mol = Chem.MolFromSmarts(pattern) if pattern else None
        queries.append((mol, mapped))
    return queries


def match_linker_hint(linker_hint: str, linker_dict: Dict[str, object]) -> Optional[str]:
    if not linker_hint:
        return None
    hint_key = normalize_linker_key(linker_hint)
    for entry in linker_dict.get("entries", []):
        mapped = entry.get("mapped")
        if not mapped:
            continue
        keys = entry.get("keys", [])
        for key in keys:
            if normalize_linker_key(key) == hint_key:
                return mapped
    return None


def match_linker_hint_entry(
    linker_hint: str,
    linker_dict: Dict[str, object],
    min_ratio: float = 0.7,
) -> Optional[Dict[str, object]]:
    if not linker_hint:
        return None
    hint_mol = Chem.MolFromSmiles(linker_hint)
    if hint_mol is None:
        return None
    hint_atoms = max(1, hint_mol.GetNumHeavyAtoms())
    best_entry = None
    best_score = 0.0
    for entry in linker_dict.get("entries", []):
        keys = entry.get("keys", [])
        for key in keys:
            key_mol = Chem.MolFromSmiles(key)
            if key_mol is None:
                continue
            try:
                res = rdFMCS.FindMCS(
                    [hint_mol, key_mol],
                    atomCompare=rdFMCS.AtomCompare.CompareElements,
                    bondCompare=rdFMCS.BondCompare.CompareAny,
                    ringMatchesRingOnly=False,
                    completeRingsOnly=False,
                    timeout=1,
                )
            except Exception:
                continue
            if res.numAtoms <= 0:
                continue
            ratio = res.numAtoms / float(hint_atoms)
            if ratio >= min_ratio and ratio > best_score:
                best_score = ratio
                best_entry = entry
    return best_entry


def infer_linker_mapping_from_hint(
    molecule: Chem.Mol,
    linker_hint: str,
    atom_owner: Dict[int, int],
    backbone_atoms: Optional[set] = None,
) -> Optional[Tuple[List[int], str]]:
    """Infer linker mapping using a raw hint SMILES (no placeholders)."""
    hint_mol = Chem.MolFromSmiles(linker_hint)
    if hint_mol is None:
        return None
    matches = molecule.GetSubstructMatches(hint_mol)
    if not matches:
        return None
    match = matches[0]
    match_set = set(match)
    if backbone_atoms and match_set.intersection(backbone_atoms):
        return None
    positions: List[int] = []
    for atom_idx in match_set:
        atom = molecule.GetAtomWithIdx(atom_idx)
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx in match_set:
                continue
            pos = atom_owner.get(nb_idx)
            if pos is not None:
                positions.append(pos)
    if len(positions) < 2:
        return None
    ordered = []
    for pos in sorted(set(positions)):
        ordered.append(pos)
    mapped = Chem.MolToSmiles(hint_mol, isomericSmiles=True)
    # replace each attachment to outside with placeholder order
    rw = Chem.RWMol(hint_mol)
    # mark attachment points: atoms that bond outside match in original molecule
    attach_atoms: List[int] = []
    for atom_idx in match_set:
        atom = molecule.GetAtomWithIdx(atom_idx)
        if any(nb.GetIdx() not in match_set for nb in atom.GetNeighbors()):
            attach_atoms.append(atom_idx)
    # map attach atoms by residue order
    attach_positions: List[Tuple[int, int]] = []
    for atom_idx in attach_atoms:
        atom = molecule.GetAtomWithIdx(atom_idx)
        pos = None
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx in match_set:
                continue
            pos = atom_owner.get(nb_idx)
            if pos is not None:
                break
        if pos is not None:
            attach_positions.append((pos, atom_idx))
    attach_positions.sort(key=lambda x: x[0])
    # create map numbers by position order
    for map_idx, (_, atom_idx) in enumerate(attach_positions, start=1):
        # find corresponding atom in hint_mol by index mapping
        # substruct match provides ordering; use the index in match to map into hint_mol
        try:
            hint_idx = match.index(atom_idx)
        except ValueError:
            continue
        a = rw.GetAtomWithIdx(hint_idx)
        a.SetAtomMapNum(map_idx)
    mapped = Chem.MolToSmiles(rw.GetMol(), isomericSmiles=True)
    return (ordered, mapped)
