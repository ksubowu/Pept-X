#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility helpers for detecting and applying disulfide (S-S) bridges."""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import rdchem

from utils import clean_smiles, get_backbone_atoms


def split_sequence_metadata(sequence: str) -> Tuple[str, List[str]]:
    """Return (core_sequence, metadata_parts) for a hyphenated sequence string."""
    if "|" not in sequence:
        return sequence.strip(), []
    head, *rest = sequence.split("|")
    metadata = [item.strip() for item in rest if item.strip()]
    return head.strip(), metadata


def merge_sequence_metadata(core: str, metadata: Sequence[str]) -> str:
    """Rebuild sequence string from core tokens and optional metadata."""
    core = core.strip()
    parts = [core] if core else []
    parts.extend(item for item in metadata if item)
    return "|".join(parts)

def _infer_seq_separator(core: str) -> str:
    """Pick '.' if present at top level, otherwise fall back to '-'."""
    depth = 0
    for ch in core:
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth = max(0, depth - 1)
        elif ch == '.' and depth == 0:
            return "."
    return "-"


def parse_disulfide_metadata(metadata_parts: Sequence[str]) -> Tuple[List[Tuple[int, int]], List[str]]:
    """
    Extract disulfide pairings from metadata entries.

    Returns (pairs, remaining_metadata).
    """
    pairs: List[Tuple[int, int]] = []
    remaining: List[str] = []
    for entry in metadata_parts:
        entry_stripped = entry.strip()
        if not entry_stripped:
            continue
        if entry_stripped.lower().startswith("s-s"):
            if ":" in entry_stripped:
                _, _, payload = entry_stripped.partition(":")
            else:
                payload = entry_stripped[3:].strip()
            for chunk in payload.replace(",", " ").split():
                if "-" not in chunk:
                    continue
                left, right = chunk.split("-", 1)
                try:
                    a = int(left)
                    b = int(right)
                except ValueError:
                    continue
                if a == b:
                    continue
                pair = tuple(sorted((a, b)))
                if pair not in pairs:
                    pairs.append(pair)
        else:
            remaining.append(entry_stripped)
    pairs.sort()
    return pairs, remaining


def format_disulfide_metadata(pairs: Sequence[Tuple[int, int]]) -> Optional[str]:
    """Create an 'S-S: a-b ...' string or None if no pairs."""
    if not pairs:
        return None
    body = " ".join(f"{a}-{b}" for a, b in pairs)
    return f"S-S: {body}"


def _assign_sulfur_to_residue(
    mol: Chem.Mol, ca_map: dict[int, int], sulfur_idx: int
) -> Optional[int]:
    """Trace from a sulfur atom to the corresponding residue index."""
    queue: deque[int] = deque([sulfur_idx])
    visited = {sulfur_idx}
    while queue:
        idx = queue.popleft()
        if idx in ca_map:
            return ca_map[idx]
        atom = mol.GetAtomWithIdx(idx)
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx in visited:
                continue
            if atom.GetAtomicNum() == 16 and nb.GetAtomicNum() == 16:
                continue  # do not cross existing disulfide bridge
            visited.add(nb_idx)
            queue.append(nb_idx)
    return None


def detect_disulfide_pairs(
    smiles: Optional[str], residue_count: int
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Locate cysteine positions and S-S pairs in a peptide SMILES.

    Parameters
    ----------
    smiles : str
        Linearised peptide SMILES.
    residue_count : int
        Number of residues expected in the backbone.
    """
    if not smiles or residue_count <= 0:
        return [], []
    clean = clean_smiles(smiles)
    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        return [], []
    matches = list(get_backbone_atoms(mol))
    if not matches or len(matches) != residue_count:
        return [], []
    ca_map = {match[1]: idx + 1 for idx, match in enumerate(matches)}
    cysteine_indices: List[int] = []
    pairs: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for bond in mol.GetBonds():
        if bond.GetBondType() != rdchem.BondType.SINGLE:
            continue
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        if a.GetAtomicNum() != 16 or b.GetAtomicNum() != 16:
            continue
        res_a = _assign_sulfur_to_residue(mol, ca_map, a.GetIdx())
        res_b = _assign_sulfur_to_residue(mol, ca_map, b.GetIdx())
        if res_a is None or res_b is None or res_a == res_b:
            continue
        pair = tuple(sorted((res_a, res_b)))
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
        for idx in pair:
            if idx not in cysteine_indices:
                cysteine_indices.append(idx)
    cysteine_indices.sort()
    pairs.sort()
    return cysteine_indices, pairs


def annotate_sequence_with_disulfides(
    sequence: str,
    n_cap_info: Optional[dict],
    c_cap_info: Optional[dict],
    cysteine_indices: List[int],
    disulfide_pairs: List[Tuple[int, int]],
) -> str:
    """Inject cysteine codes and S-S metadata into a sequence string."""
    if not cysteine_indices and not disulfide_pairs:
        return sequence
    core, metadata = split_sequence_metadata(sequence)
    _, metadata = parse_disulfide_metadata(metadata)
    sep = _infer_seq_separator(core)
    tokens = core.split(sep) if core else []
    n_cap_code = (n_cap_info or {}).get("code")
    start_offset = 1 if n_cap_code and n_cap_code not in {"", "H"} else 0
    for res_idx in cysteine_indices:
        token_pos = start_offset + res_idx - 1
        if 0 <= token_pos < len(tokens):
            tokens[token_pos] = "C"
    core = sep.join(tokens)
    ss_meta = format_disulfide_metadata(disulfide_pairs)
    if ss_meta:
        metadata = metadata + [ss_meta]
    return merge_sequence_metadata(core, metadata)


def extract_disulfide_pairs_from_sequence(
    sequence: str,
) -> Tuple[str, List[Tuple[int, int]], List[str]]:
    """
    Split sequence into (core, disulfide_pairs, other_metadata_entries).
    """
    core, metadata = split_sequence_metadata(sequence)
    pairs, remaining = parse_disulfide_metadata(metadata)
    return core, pairs, remaining


def _remove_thiol_hydrogens(rw: Chem.RWMol, atom_idx: int) -> None:
    """Delete explicit hydrogens attached to the provided atom index."""
    atom = rw.GetAtomWithIdx(atom_idx)
    hydrogens = [nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1]
    for h_idx in sorted(hydrogens, reverse=True):
        if h_idx < rw.GetNumAtoms():
            rw.RemoveAtom(h_idx)
    atom = rw.GetAtomWithIdx(atom_idx)
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(0)


def apply_disulfide_bonds(
    mol: Chem.Mol,
    residue_pairs: Sequence[Tuple[int, int]],
) -> Chem.Mol:
    """Return a copy of `mol` with specified residue pairs connected via S-S bonds."""
    if not residue_pairs:
        return Chem.Mol(mol)
    matches = list(get_backbone_atoms(mol))
    if not matches:
        raise ValueError("Unable to locate peptide backbone while adding S-S bonds.")
    ca_map = {match[1]: idx + 1 for idx, match in enumerate(matches)}
    residue_to_sulfur: dict[int, int] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 16:
            continue
        res_idx = _assign_sulfur_to_residue(mol, ca_map, atom.GetIdx())
        if res_idx is None:
            continue
        residue_to_sulfur.setdefault(res_idx, atom.GetIdx())
    rw = Chem.RWMol(mol)
    seen_pairs: set[Tuple[int, int]] = set()
    for a, b in residue_pairs:
        if a == b:
            continue
        pair = tuple(sorted((a, b)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        s1 = residue_to_sulfur.get(pair[0])
        s2 = residue_to_sulfur.get(pair[1])
        if s1 is None or s2 is None:
            raise ValueError(f"Unable to locate sulfur atoms for residues {pair}.")
        _remove_thiol_hydrogens(rw, s1)
        _remove_thiol_hydrogens(rw, s2)
        if rw.GetBondBetweenAtoms(s1, s2) is None:
            rw.AddBond(s1, s2, rdchem.BondType.SINGLE)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def apply_disulfide_bonds_with_map(
    mol: Chem.Mol,
    residue_pairs: Sequence[Tuple[int, int]],
    residue_to_sulfur: dict[int, int],
) -> Chem.Mol:
    """Apply S-S bonds using a precomputed residue->sulfur atom index map."""
    if not residue_pairs:
        return Chem.Mol(mol)
    rw = Chem.RWMol(mol)
    seen_pairs: set[Tuple[int, int]] = set()
    for a, b in residue_pairs:
        if a == b:
            continue
        pair = tuple(sorted((a, b)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        s1 = residue_to_sulfur.get(pair[0])
        s2 = residue_to_sulfur.get(pair[1])
        if s1 is None or s2 is None:
            raise ValueError(f"Unable to locate sulfur atoms for residues {pair}.")
        _remove_thiol_hydrogens(rw, s1)
        _remove_thiol_hydrogens(rw, s2)
        if rw.GetBondBetweenAtoms(s1, s2) is None:
            rw.AddBond(s1, s2, rdchem.BondType.SINGLE)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out
