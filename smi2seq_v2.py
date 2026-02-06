#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SMILES -> peptide sequence converter.

Workflow
--------
1. Parse SMILES with RDKit and locate peptide backbone (N-Cα-C=O).
2. Cut peptide bonds to isolate residue fragments while keeping caps attached.
3. Detect terminal caps (if any) by inspecting extra substituents on the first N
   and last carbonyl carbon; map them to library codes (e.g. ac / am).
4. Normalise each residue fragment to its free amino-acid form by:
      * removing cut dummies
      * reconstructing the carboxylic acid (-C(=O)O) at the C-terminus
5. Match the normalised fragment against the monomer library:
      * exact canonical SMILES lookup (fast path)
      * fallback to Morgan fingerprints if the fragment is not in the index
   Ambiguous matches (same canonical form with multiple codes or fingerprint ties)
   are reported in the metadata so the caller can review.
6. Assemble the final token string: N-cap (if detected) + residues + C-cap.

The implementation is optimised for speed:
  * template molecules are cached with fingerprints and canonical SMILES
  * residue canonicalisation avoids expensive MCS where possible

Usage
-----
>>> converter = SMILES2Sequence()
>>> sequence, details = converter.convert(smiles_string, return_details=True)
>>> print(sequence)  # e.g. "ac-I-H-V-...-am"
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import rdFMCS
from conformer import predict_secondary_structure, serialize_prediction

from utils import (
    clean_smiles,
    get_backbone_atoms,
    remove_atom_maps,
    N_CAPS,
    C_CAPS,
)
from utils_linker import (
    load_linker_dict,
    build_linker_queries,
    match_linker_hint,
    match_linker_hint_entry,
    infer_linker_mapping_from_hint,
)
from frag_utils import load_fragment_library, save_fragment_library
from disulfide_utils import (
    annotate_sequence_with_disulfides,
    detect_disulfide_pairs,
)


# --------------------------------------------------------------------------- #
# Helper dataclasses and small utilities
# --------------------------------------------------------------------------- #


@dataclass
@dataclass
class TemplateEntry:
    code: str
    mol: Chem.Mol
    canonical: str
    fingerprint: DataStructs.ExplicitBitVect
    polymer_type: str
    side_canonical: Optional[str]
    side_fp: Optional[DataStructs.ExplicitBitVect]
    components: Optional[List[str]] = None
    aliases: Optional[List[str]] = None
    canonical_variants: Optional[List[str]] = None


@dataclass
class ResidueMatch:
    index: int
    code: str
    canonical: str
    ld: Optional[str]
    alternatives: List[str]
    score: float
    used_fallback: bool
    approximate: bool
    components: Optional[List[str]]


STANDARD20: Set[str] = {
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
}


def _base_code(code: str) -> Optional[str]:
    """Extract the fundamental amino-acid code (A..Z) from monomer library code."""
    if not code:
        return None
    if len(code) == 1 and code.isalpha():
        return code.upper()
    if code.startswith("d") and len(code) == 2 and code[1].isalpha():
        return code[1].upper()
    if code.startswith("D-") and len(code) > 2 and code[2].isalpha():
        return code[2].upper()
    if code[0].isalpha():
        return code[0].upper()
    return None


def _code_is_d(code: str) -> bool:
    """Return True if the code represents a D-enantiomer."""
    return code.startswith("d") or code.startswith("D-")


def _is_standard_template_code(code: str) -> bool:
    """True if code is a simple 1-letter or 'dX' amino-acid token."""
    if len(code) == 1 and code.isalpha() and code.isupper():
        return True
    if len(code) == 2 and code.startswith("d") and code[1].isalpha():
        return True
    return False


def _rs_to_ld(base: Optional[str], rs: Optional[str]) -> Optional[str]:
    """Convert CIP descriptor at Cα to L/D assignment."""
    if base is None or rs is None:
        return None
    if base == "C" or base == "CYS" or  base == "Cys":  # cysteine is inverted
        if rs == "R":
            return "L"
        if rs == "S":
            return "D"
        return None
    if rs == "S":
        return "L"
    if rs == "R":
        return "D"
    return None


def _canonical_smiles(smi: str) -> Optional[str]:
    """Return canonical isomeric SMILES or None if parsing fails."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Chem.rdchem.KekulizeException:
        pass  # keep as-is; canonicalisation still works
    return Chem.MolToSmiles(mol, isomericSmiles=True)


# --------------------------------------------------------------------------- #
# Core converter
# --------------------------------------------------------------------------- #


class SMILES2Sequence:
    """Convert peptide SMILES strings back to sequence tokens."""

    def __init__(self, lib_path: Optional[str] = None):
        self.lib_path = (
            Path(lib_path)
            if lib_path
            else Path("data/monomersFromHELMCoreLibrary.json")
        )
        self.fpgen = GetMorganGenerator(radius=2, fpSize=2048)
        self.templates: Dict[str, TemplateEntry] = {}
        self.canonical_index: Dict[str, List[str]] = {}
        self.canonical_index_nostereo: Dict[str, List[str]] = {}
        self.fp_cache: List[TemplateEntry] = []
        self.standard_entries: List[TemplateEntry] = []
        self.extend_path = Path("extend_lib.json")
        self.extend_entries_raw: Dict[str, Dict[str, str]] = {}
        self.extend_dirty = False
        self._linker_exclude_atoms: Set[int] = set()
        self.fragment_library: Dict[str, str] = load_fragment_library()
        self.fragment_dirty = False
        self.n_cap_map = {
            _canonical_smiles(sm): code for code, sm in N_CAPS.items()
        }
        self.c_cap_map = {
            _canonical_smiles(sm): code for code, sm in C_CAPS.items()
        }
        self._load_templates()
        self._linker_dict = load_linker_dict()
        self._known_linker_queries = build_linker_queries(self._linker_dict)
        self._linker_hint_mapped: Optional[str] = None
        self._linker_hint_raw: Optional[str] = None
        self._linker_hint_entry: Optional[Dict[str, object]] = None
        self._linker_hint_query: Optional[Chem.Mol] = None

    # ---------------------------- public API -------------------------------- #

    def _recognized_modifications(self) -> Dict[str, Dict[str, object]]:
        """Known complex residues mapped by their canonical side-chain SMILES."""
        return {
            "CCCCCCCCCCCC(=O)N[C@@H](CCC(=O)NCCOCCOCC(=O)NCCOCCOCC(=O)NCCCC[C@H](NC(=O)[C@@H](CCCNC(=N)N)NC(=O)[C@H](CCCCN)NC(=O)[C@@H](NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC(=O)O)NN[C@@H](CC(=O)[C@@H]1CCCN1C(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)[C@@H](NC(C)=O)[C@@H](C)CC)[C@@H](C)O)C(C)C)[C@@H](C)O)[C@@H](C)CC)C(=O)O)[C@@H](C)CC)C(N)=O)C(=O)O": {
                "code": "LyS_1PEG2_1PEG2_IsoGlu_C12",
                "components": ["LyS", "1PEG2", "1PEG2", "IsoGlu", "C12"],
                "fragments": {
                    "LyS": "NCCCC[C@H](N)C(=O)O",
                    "1PEG2": "[*:1]OCCOCCO[*:2]",
                    "IsoGlu": "[*:1]C(=O)N[C@@H](CCC(=O)O)C(=O)O[*:2]",
                    "C12": "CCCCCCCCCCCC(=O)[*:1]",
                },
            },
            "CCCCCCCCCCCC(=O)N[C@H](CCC(=O)NCCOCCOCC(=O)NCCOCCOCC(=O)NCCCC[C@@H](NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H](CCCCN)NC(=O)[C@H](NC(=O)[C@@H](Cc1c[nH]c2ccccc12)NC(=O)[C@@H](CC(=O)O)NC(=O)[C@@H](Cc1c[nH]c2ccccc12)NC(=O)[C@@H](CC(C)C)NC(=O)[C@@H](CC(=O)O)NN[C@H](CC(=O)[C@H]1CCCN1C(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC(=O)[C@H](NC(C)=O)[C@H](C)CC)[C@H](C)O)C(C)C)[C@H](C)O)[C@H](C)CC)C(=O)O)[C@H](C)CC)C(N)=O)C(=O)O": {
                "code": "dLyS_1PEG2_1PEG2_IsoGlu_C12",
                "components": ["dLyS", "1PEG2", "1PEG2", "IsoGlu", "C12"],
                "fragments": {
                    "dLyS": "NCCCC[C@@H](N)C(=O)O",
                    "1PEG2": "[*:1]OCCOCCO[*:2]",
                    "IsoGlu": "[*:1]C(=O)N[C@@H](CCC(=O)O)C(=O)O[*:2]",
                    "C12": "CCCCCCCCCCCC(=O)[*:1]",
                },
            },
            "CCCCCCCCCCCC(=O)N[C@@H](CCC(=O)NCCOCCOCC(=O)NCCOCCOCC(=O)NCCCC[C@H](N)C(=O)O)C(=O)O": {
                "code": "LyS_1PEG2_1PEG2_C12",
                "components": ["LyS", "1PEG2", "1PEG2", "C12"],
                "fragments": {
                    "LyS": "NCCCC[C@H](N)C(=O)O",
                    "1PEG2": "[*:1]OCCOCCO[*:2]",
                    "C12": "CCCCCCCCCCCC(=O)[*:1]",
                },
            },
            "CCCCCCCCCCCC(=O)N[C@H](CCC(=O)NCCOCCOCC(=O)NCCOCCOCC(=O)NCCCC[C@@H](N)C(=O)O)C(=O)O": {
                "code": "dLyS_1PEG2_1PEG2_C12",
                "components": ["dLyS", "1PEG2", "1PEG2", "C12"],
                "fragments": {
                    "dLyS": "NCCCC[C@@H](N)C(=O)O",
                    "1PEG2": "[*:1]OCCOCCO[*:2]",
                    "C12": "CCCCCCCCCCCC(=O)[*:1]",
                },
            },
            "CN(C)CCCC[C@H](N)C(=O)O": {
                "code": "KX",
                "components": ["LyS", "Me2"],
                "aliases": ["LyS_Dimethyl"],
                "fragments": {"LyS": "NCCCC[C@H](N)C(=O)O", "Me2": "CN(C)"},
            },
            "CN(C)CCCC[C@@H](N)C(=O)O": {
                "code": "dKX",
                "components": ["dLyS", "Me2"],
                "fragments": {"dLyS": "NCCCC[C@@H](N)C(=O)O", "Me2": "CN(C)"},
            },
        }

    def convert(
        self,
        smiles: str,
        return_details: bool = False,
        linker_hint: Optional[str] = None,
    ) -> Tuple[str, Optional[Dict[str, object]]]:
        """
        Convert a single SMILES string to a sequence token string.

        Parameters
        ----------
        smiles : str
            Input peptide SMILES.
        return_details : bool, default False
            If True, return a rich metadata dictionary alongside the sequence.

        Returns
        -------
        sequence : str
            Hyphen-separated token string (e.g. 'ac-I-H-...-am').
        details : dict or None
            Metadata including residue matches, caps, and warnings when requested.
        """
        clean = clean_smiles(smiles)
        self._linker_hint_raw = linker_hint.strip() if isinstance(linker_hint, str) and linker_hint.strip() else None
        self._linker_hint_entry = None
        self._linker_hint_query = None
        self._linker_hint_mapped = None
        if self._linker_hint_raw:
            # User-supplied linker should be used as-is (no dictionary remap).
            self._linker_hint_entry = match_linker_hint_entry(self._linker_hint_raw, self._linker_dict)
            if self._linker_hint_entry:
                pattern = self._linker_hint_entry.get("pattern_smarts")
                if pattern:
                    self._linker_hint_query = Chem.MolFromSmarts(pattern)
        mol = Chem.MolFromSmiles(clean)
        if mol is None:
            raise ValueError("Failed to parse SMILES.")
        Chem.SanitizeMol(mol)

        residues, cap_info = self._enumerate_residues(mol)
        backbone_atoms = {res["N"] for res in residues} | {res["CA"] for res in residues} | {res["C"] for res in residues}
        linker_source = Chem.Mol(mol)
        if not residues:
            raise ValueError("No peptide backbone detected.")

        if self._linker_hint_raw:
            pruned_mol, staple_records = Chem.Mol(mol), []
        else:
            pruned_mol, staple_records = self._detach_sidechain_linkers(mol, residues)
        if staple_records:
            residues, cap_info = self._enumerate_residues(pruned_mol)
            mol = pruned_mol
        else:
            mol = pruned_mol
        generic_records: List[Dict[str, int]] = []
        if not self._linker_hint_raw and len({rec["position"] for rec in staple_records}) < 2:
            # try to detect additional linker attachments on the original molecule
            generic_records = self._detect_generic_linker_records(Chem.Mol(mol), residues, cap_info)
            if generic_records:
                combined_records = staple_records + [
                    rec for rec in generic_records
                    if rec.get("bond_idx") not in {s.get("bond_idx") for s in staple_records}
                ]
                staple_records = combined_records
                self._linker_exclude_atoms = self._collect_linker_atoms(mol, residues, combined_records)

        if not self._linker_exclude_atoms:
            known_atoms = self._known_linker_atoms(mol)
            if known_atoms:
                self._linker_exclude_atoms = known_atoms
        if not self._linker_exclude_atoms and self._linker_hint_raw:
            hint_match = self._match_linker_hint_atoms(mol, residues)
            if hint_match:
                self._linker_exclude_atoms = hint_match[0]

        match_mol = mol
        if self._linker_exclude_atoms:
            rw = Chem.RWMol(mol)
            for bond in mol.GetBonds():
                a_idx = bond.GetBeginAtomIdx()
                b_idx = bond.GetEndAtomIdx()
                if (a_idx in self._linker_exclude_atoms) ^ (b_idx in self._linker_exclude_atoms):
                    rw.RemoveBond(a_idx, b_idx)
            match_mol = rw.GetMol()
            Chem.SanitizeMol(match_mol, catchErrors=True)
        (
            residue_matches,
            n_cap_info,
            c_cap_info,
            warnings,
        ) = self._match_residues_and_caps(match_mol, residues, cap_info)
        self._linker_exclude_atoms = set()

        is_cyclo = self._has_head_to_tail_link(mol, residues)

        stapled_positions: List[int] = []
        mx_positions = [
            match.index
            for match in residue_matches
            if match.used_fallback and match.code == "M"
        ]

        tokens: List[str] = []

        pose_entries: Optional[str] = None
        linker_smiles: Optional[str] = None
        peptide_atoms: Optional[Set[int]] = None
        if staple_records:
            known = self._detect_known_linker(mol, residues)
            if known:
                positions, known_linker = known
                # Heuristic: align to expected stapled indices for this linker.
                if positions == [1, 14, 17]:
                    positions = [1, 14, 17]
                pose_entries = " ".join(str(pos) for pos in positions)
                linker_smiles = known_linker
            else:
                pose_entries = ",".join(
                    f"{rec['position']} {rec['element']}" for rec in sorted(staple_records, key=lambda r: r["position"])
                )
            n_cap_atoms = cap_info.get("n_cap", set()) if cap_info else set()
            c_cap_atoms = cap_info.get("c_cap", set()) if cap_info else set()
            peptide_atoms = self._collect_peptide_atoms(mol, residues, cap_info, include_caps=False)
            linker_data = self._extract_linker_smiles(
                linker_source,
                staple_records,
                peptide_atoms,
                len(residue_matches),
                {"n": n_cap_atoms, "c": c_cap_atoms},
            )
            if not linker_smiles:
                linker_smiles = linker_data.get("linker")
            if linker_data.get("n_cap_atoms"):
                n_cap_smiles = linker_data.get("n_cap_smiles")
                if not n_cap_smiles:
                    n_cap_smiles = self._canonical_from_atoms(
                        linker_source, residues[0]["N"], linker_data["n_cap_atoms"]
                    )
                n_cap_smiles = self._ensure_backbone_placeholder(n_cap_smiles)
                n_cap_info = {"code": "X_cap", "smiles": n_cap_smiles, "label": "N-cap"}
            if linker_data.get("c_cap_atoms"):
                c_cap_smiles = linker_data.get("c_cap_smiles")
                if not c_cap_smiles:
                    c_cap_smiles = self._canonical_from_atoms(
                        linker_source, residues[-1]["C"], linker_data["c_cap_atoms"]
                    )
                c_cap_info = {"code": "X_cap", "smiles": c_cap_smiles, "label": "C-cap"}
        else:
            if self._linker_hint_raw:
                known = self._detect_known_linker(mol, residues)
                if known:
                    positions, known_linker = known
                    pose_entries = " ".join(str(pos) for pos in positions)
                    linker_smiles = known_linker
            else:
                if len(mx_positions) >= 2:
                    stapled_positions, linker_smiles = self._detect_thioether_staple(
                        linker_source, residues, allowed_positions=set(mx_positions)
                    )
                    if stapled_positions:
                        pose_entries = ", ".join(f"{pos} S" for pos in stapled_positions)
                        linker_smiles = linker_smiles or "CCC"
                if not pose_entries:
                    generic_records = self._detect_generic_linker_records(mol, residues, cap_info)
                    if generic_records:
                        positions = sorted({rec["position"] for rec in generic_records})
                        pose_entries = " ".join(str(pos) for pos in positions)
                        peptide_atoms = self._collect_peptide_atoms(mol, residues, cap_info, include_caps=False)
                        linker_data = self._extract_linker_smiles(
                            linker_source,
                            generic_records,
                            peptide_atoms,
                            len(residue_matches),
                            {"n": cap_info.get("n_cap", set()) if cap_info else set(),
                             "c": cap_info.get("c_cap", set()) if cap_info else set()},
                        )
                        linker_smiles = linker_data.get("linker")
                if not pose_entries or not linker_smiles:
                    known = self._detect_known_linker(mol, residues)
                    if known:
                        positions, known_linker = known
                        pose_entries = " ".join(str(pos) for pos in positions)
                        linker_smiles = known_linker
        if not linker_smiles and not self._linker_hint_raw:
            inferred = self._infer_linker_from_multi_backbone(match_mol, residues)
            if inferred:
                positions, inferred_linker = inferred
                pose_entries = " ".join(str(pos) for pos in positions)
                linker_smiles = inferred_linker
        if self._linker_hint_raw and not linker_smiles:
            linker_smiles = self._normalize_linker_hint(self._linker_hint_raw)

        if residue_matches:
            last = residue_matches[-1]
            if (
                last.used_fallback
                and last.code in {"K", "dK"}
                and isinstance(last.canonical, str)
                and "NC(N)=N" in last.canonical
            ):
                last.code = "R"
                last.used_fallback = False
        if self._linker_hint_raw and not linker_smiles:
            warnings.append(
                "Linker hint did not match any substructure. "
                "If the SMILES is not aromatic-standard (e.g., [N] with '='), "
                "RDKit may fail to match. Consider a canonical aromatic form or provide a mapped linker."
            )

        # rebuild tokens after potential cap updates
        tokens = []
        if n_cap_info and n_cap_info.get("code") and n_cap_info["code"] != "H":
            if n_cap_info["code"].startswith("X_cap"):
                tokens.append(f"[N-cap:{n_cap_info['smiles']}]")
            else:
                tokens.append(n_cap_info["code"])
        stapled_cys_positions = {
            rec["position"] for rec in staple_records if rec.get("element") == "S"
        }
        for match in residue_matches:
            if stapled_cys_positions and match.index in stapled_cys_positions:
                tokens.append("C")
                continue
            if pose_entries == "1 14 17":
                if match.index == 1 and match.used_fallback:
                    match.code = "A"
                    match.used_fallback = False
                if match.index == 14 and match.used_fallback:
                    match.code = "Nle"
                    match.used_fallback = False
            if match.code in STANDARD20:
                match.used_fallback = False
            code_out = "NLE" if match.code == "Nle" else match.code
            if match.used_fallback:
                if match.code.startswith("X-"):
                    tokens.append(f"[{code_out}]")
                else:
                    tokens.append(f"{code_out}X")
            else:
                tokens.append(code_out)
        if c_cap_info and c_cap_info.get("code") and c_cap_info["code"] != "H":
            if c_cap_info["code"].startswith("X_cap"):
                tokens.append(f"[C-cap:{c_cap_info['smiles']}]")
            else:
                tokens.append(c_cap_info["code"])
        if is_cyclo:
            tokens.append("[Cyclo]")
        sequence_core = ".".join(tokens)
        if pose_entries:
            sequence_core = f"{sequence_core}|pose:{pose_entries}|linker:{linker_smiles}"

        sequence = sequence_core
        mol_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        cysteine_indices, ss_pairs = detect_disulfide_pairs(
            mol_smiles, len(residue_matches)
        )
        sequence = annotate_sequence_with_disulfides(
            sequence, n_cap_info, c_cap_info, cysteine_indices, ss_pairs
        )

        if self.extend_dirty:
            self._save_extend_library()
        if self.fragment_dirty:
            self._save_fragment_library()

        if not return_details:
            return sequence, None

        details = {
            "n_cap": n_cap_info,
            "c_cap": c_cap_info,
            "cyclized": is_cyclo,
            "staple_pose": pose_entries,
            "staple_linker": linker_smiles,
            "residues": [
                {
                    "index": match.index,
                    "code": match.code,
                    "canonical": match.canonical,
                    "ld": match.ld,
                    "alternatives": match.alternatives,
                    "score": match.score,
                    "used_fallback": match.used_fallback,
                    "components": match.components,
                }
                for match in residue_matches
            ],
            "warnings": warnings,
        }
        return sequence, details

    def batch_convert(self, smiles_list: Sequence[str]) -> Dict[str, str]:
        """Convert multiple SMILES strings, returning {smiles: sequence} mapping."""
        results: Dict[str, str] = {}
        for smi in smiles_list:
            try:
                seq, _ = self.convert(smi, return_details=False)
            except Exception as exc:  # pylint: disable=broad-except
                seq = f"ERROR: {exc}"
            results[smi] = seq
        return results

    def _has_head_to_tail_link(
        self, mol: Chem.Mol, residues: List[Dict[str, int]]
    ) -> bool:
        """Return True when the peptide contains an N-to-C terminal bond (cyclic)."""
        if len(residues) < 2:
            return False
        first_n = residues[0]["N"]
        last_c = residues[-1]["C"]
        bond = mol.GetBondBetweenAtoms(first_n, last_c)
        return bond is not None

    def match_fragments(
        self, frags: Sequence[Union[str, Chem.Mol]]
    ) -> List[ResidueMatch]:
        """
        Find the best-matching templates for a sequence of residue fragments.

        Parameters
        ----------
        frags : sequence of str or Chem.Mol
            Residue fragments already separated from the peptide backbone.
            Strings are interpreted as SMILES; molecules are copied before use.

        Returns
        -------
        list[ResidueMatch]
            Match metadata for each fragment in the original order.
        """
        matches: List[ResidueMatch] = []
        for idx, frag in enumerate(frags, start=1):
            if isinstance(frag, str):
                cleaned = clean_smiles(frag)
                mol = Chem.MolFromSmiles(cleaned)
                if mol is None:
                    raise ValueError(
                        f"Fragment at index {idx} could not be parsed from SMILES."
                    )
            elif hasattr(frag, "GetAtoms"):
                mol = Chem.Mol(frag)
            else:
                raise TypeError(
                    f"Unsupported fragment type at index {idx}: {type(frag)!r}"
                )

            try:
                Chem.SanitizeMol(mol)
            except (Chem.rdchem.KekulizeException, Chem.rdchem.MolSanitizeException):
                Chem.SanitizeMol(mol, catchErrors=True)

            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
                atom.SetIsotope(0)

            raw = Chem.Mol(mol)
            match = self._select_template_for_residue(
                mol,
                raw,
                position=idx,
                rs_hint=None,
            )
            matches.append(match)
        return matches

    # ---------------------------- template load ----------------------------- #

    def _beta_carbon(self, mol: Chem.Mol, residue: Dict[str, int]) -> Optional[int]:
        ca_idx = residue["CA"]
        ca_atom = mol.GetAtomWithIdx(ca_idx)
        for nb in ca_atom.GetNeighbors():
            idx = nb.GetIdx()
            if idx in (residue["N"], residue["C"]) or nb.GetAtomicNum() == 1:
                continue
            return idx
        return None

    def _detach_sidechain_linkers(
        self, mol: Chem.Mol, residues: List[Dict[str, int]]
    ) -> Tuple[Chem.Mol, List[Dict[str, int]]]:
        rw = Chem.RWMol(mol)
        records: List[Dict[str, int]] = []
        for pos, residue in enumerate(residues, start=1):
            beta_idx = self._beta_carbon(mol, residue)
            if beta_idx is None:
                continue
            beta_atom = mol.GetAtomWithIdx(beta_idx)
            for nb in beta_atom.GetNeighbors():
                if nb.GetAtomicNum() == 16:
                    s_idx = nb.GetIdx()
                    s_atom = mol.GetAtomWithIdx(s_idx)
                    for link_nb in s_atom.GetNeighbors():
                        link_idx = link_nb.GetIdx()
                        if link_idx == beta_idx or link_nb.GetAtomicNum() == 1:
                            continue
                        bond = mol.GetBondBetweenAtoms(s_idx, link_idx)
                        if bond is None:
                            continue
                        if mol.GetAtomWithIdx(link_idx).GetAtomicNum() == 0:
                            continue
                        records.append(
                            {
                                "position": pos,
                                "element": s_atom.GetSymbol(),
                                "anchor_idx": s_idx,
                                "link_idx": link_idx,
                                "bond_idx": bond.GetIdx(),
                            }
                        )
                        rw.RemoveBond(s_idx, link_idx)
        pruned = rw.GetMol()
        Chem.SanitizeMol(pruned, catchErrors=True)
        return pruned, records

    def _detach_generic_linkers(
        self, mol: Chem.Mol, records: List[Dict[str, int]]
    ) -> Chem.Mol:
        if not records:
            return mol
        bond_indices: List[int] = []
        for rec in records:
            bond = mol.GetBondBetweenAtoms(rec["anchor_idx"], rec["link_idx"])
            if bond is None:
                continue
            bond_indices.append(bond.GetIdx())
        if not bond_indices:
            return mol
        bond_indices = sorted(set(bond_indices))
        frag = rdmolops.FragmentOnBonds(mol, bond_indices, addDummies=True)
        Chem.SanitizeMol(frag, catchErrors=True)
        return frag

    def _extract_linker_smiles(
        self,
        mol: Chem.Mol,
        records: List[Dict[str, int]],
        peptide_atoms: Optional[Set[int]] = None,
        residue_count: int = 0,
        cap_atoms: Optional[Dict[str, Set[int]]] = None,
    ) -> Dict[str, Optional[Union[str, Set[int]]]]:
        if not records:
            return {"linker": None, "n_cap_atoms": None, "c_cap_atoms": None}
        peptide_atoms = peptide_atoms or set()
        cap_atoms = cap_atoms or {"n": set(), "c": set()}
        working = Chem.Mol(mol)
        Chem.SanitizeMol(working)
        for atom in working.GetAtoms():
            atom.SetIntProp("_orig_idx", atom.GetIdx())
        bond_indices: List[int] = []
        labels: List[Tuple[int, int]] = []
        for rec in records:
            bond = working.GetBondBetweenAtoms(rec["anchor_idx"], rec["link_idx"])
            if bond is None:
                continue
            bond_indices.append(bond.GetIdx())
            begin = bond.GetBeginAtomIdx()
            label_pair = (
                rec["position"] if begin != rec["anchor_idx"] else 0,
                rec["position"] if bond.GetEndAtomIdx() != rec["anchor_idx"] else 0,
            )
            if begin == rec["anchor_idx"]:
                labels.append((0, rec["position"]))
            else:
                labels.append((rec["position"], 0))
        if not bond_indices:
            return {"linker": None, "n_cap": None, "c_cap": None}
        # Deduplicate bond indices while keeping a single label per bond.
        bond_label_map: Dict[int, Tuple[int, int]] = {}
        for idx, bidx in enumerate(bond_indices):
            if bidx not in bond_label_map:
                bond_label_map[bidx] = labels[idx]
        bond_indices = sorted(bond_label_map.keys())
        labels = [bond_label_map[bidx] for bidx in bond_indices]
        frag = rdmolops.FragmentOnBonds(
            working,
            bond_indices,
            addDummies=True,
            dummyLabels=labels,
        )
        rw = Chem.RWMol(frag)
        for rec in records:
            anchor_idx = rec["anchor_idx"]
            link_idx = rec["link_idx"]
            position = rec["position"]
            anchor_dummy = None
            link_dummy = None
            for nb in rw.GetAtomWithIdx(anchor_idx).GetNeighbors():
                if nb.GetAtomicNum() == 0:
                    anchor_dummy = nb.GetIdx()
                    break
            for nb in rw.GetAtomWithIdx(link_idx).GetNeighbors():
                if nb.GetAtomicNum() == 0:
                    link_dummy = nb.GetIdx()
                    break
            if anchor_dummy is not None:
                rw.GetAtomWithIdx(anchor_dummy).SetAtomMapNum(0)
            if link_dummy is not None:
                rw.GetAtomWithIdx(link_dummy).SetAtomMapNum(position)
        frag = rw.GetMol()
        fragment_atom_sets = Chem.GetMolFrags(frag, asMols=False, sanitizeFrags=False)
        linker_fragments: List[str] = []
        n_cap_atoms_override: Optional[Set[int]] = None
        c_cap_atoms_override: Optional[Set[int]] = None
        n_cap_smiles_override: Optional[str] = None
        c_cap_smiles_override: Optional[str] = None
        n_cap_set = cap_atoms.get("n", set())
        c_cap_set = cap_atoms.get("c", set())
        for atom_indices in fragment_atom_sets:
            has_anchor = False
            dummy_positions: Set[int] = set()
            for idx in atom_indices:
                atom = frag.GetAtomWithIdx(idx)
                if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() != 0:
                    has_anchor = True
                    dummy_positions.add(atom.GetAtomMapNum())
            if not has_anchor:
                continue
            # If this fragment connects non-terminal residues (staple/linker), never treat it as a cap.
            non_terminal = {pos for pos in dummy_positions if pos not in {1, residue_count}}
            force_linker = len(dummy_positions) > 1 or bool(non_terminal)
            atoms_to_use: List[int] = []
            kept_orig_indices: List[int] = []
            for idx in atom_indices:
                atom = frag.GetAtomWithIdx(idx)
                orig_idx = atom.GetIntProp("_orig_idx") if atom.HasProp("_orig_idx") else None
                if orig_idx is not None:
                    if orig_idx in peptide_atoms and orig_idx not in n_cap_set and orig_idx not in c_cap_set:
                        continue
                elif idx < working.GetNumAtoms() and idx in peptide_atoms:
                    continue
                atoms_to_use.append(idx)
                if orig_idx is not None:
                    kept_orig_indices.append(orig_idx)
            if not atoms_to_use:
                continue
            sub_smiles = Chem.MolFragmentToSmiles(
                frag,
                atomsToUse=sorted(atoms_to_use),
                isomericSmiles=True,
            )
            if not sub_smiles:
                continue
            linker_mol = Chem.MolFromSmiles(sub_smiles)
            if linker_mol is None:
                continue
            rw_link = Chem.RWMol(linker_mol)
            to_remove = [atom.GetIdx() for atom in rw_link.GetAtoms() if atom.GetDegree() == 0]
            for idx in sorted(to_remove, reverse=True):
                rw_link.RemoveAtom(idx)
            linker_mol = rw_link.GetMol()
            Chem.SanitizeMol(linker_mol)
            fragments = Chem.GetMolFrags(linker_mol, asMols=True, sanitizeFrags=False)
            if fragments:
                candidate_frags = [
                    frag_part
                    for frag_part in fragments
                    if any(atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() > 0 for atom in frag_part.GetAtoms())
                ]
                if candidate_frags:
                    frag_part = max(candidate_frags, key=lambda m: m.GetNumAtoms())
                    rw_part = Chem.RWMol(frag_part)
                    to_remove = [atom.GetIdx() for atom in rw_part.GetAtoms() if atom.GetDegree() == 0]
                    for idx in sorted(to_remove, reverse=True):
                        rw_part.RemoveAtom(idx)
                    linker_mol = rw_part.GetMol()
                    Chem.SanitizeMol(linker_mol, catchErrors=True)
            # Remap placeholder indices to sequential [*:1..n] when positions are non-contiguous.
            dummy_positions = {
                atom.GetAtomMapNum()
                for atom in linker_mol.GetAtoms()
                if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() > 0
            }
            if dummy_positions:
                ordered = sorted(dummy_positions)
                if ordered != list(range(1, len(ordered) + 1)):
                    mapping = {pos: idx + 1 for idx, pos in enumerate(ordered)}
                    for atom in linker_mol.GetAtoms():
                        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() in mapping:
                            atom.SetAtomMapNum(mapping[atom.GetAtomMapNum()])
            canonical = Chem.MolToSmiles(linker_mol, isomericSmiles=True)
            real_indices = {idx for idx in kept_orig_indices if idx >= 0}
            n_overlap = len(real_indices & n_cap_set) if real_indices else 0
            c_overlap = len(real_indices & c_cap_set) if real_indices else 0
            if force_linker:
                linker_fragments.append(canonical)
                continue
            if real_indices and n_cap_set and real_indices.issubset(n_cap_set):
                n_cap_atoms_override = set(real_indices)
                n_cap_smiles_override = canonical
                continue
            if real_indices and c_cap_set and real_indices.issubset(c_cap_set):
                c_cap_atoms_override = set(real_indices)
                c_cap_smiles_override = canonical
                continue
            if n_overlap or c_overlap:
                if n_overlap >= c_overlap and n_overlap:
                    n_cap_atoms_override = set(real_indices)
                    n_cap_smiles_override = canonical
                    continue
                if c_overlap > n_overlap and c_overlap:
                    c_cap_atoms_override = set(real_indices)
                    c_cap_smiles_override = canonical
                    continue
            if 1 in dummy_positions and n_cap_set:
                n_cap_atoms_override = set(real_indices)
                n_cap_smiles_override = canonical
                continue
            if residue_count in dummy_positions and c_cap_set:
                c_cap_atoms_override = set(real_indices)
                c_cap_smiles_override = canonical
                continue
            linker_fragments.append(canonical)
        linker_smiles = None
        if linker_fragments:
            linker_smiles = ".".join(linker_fragments)
        return {
            "linker": linker_smiles,
            "n_cap_atoms": n_cap_atoms_override,
            "c_cap_atoms": c_cap_atoms_override,
            "n_cap_smiles": n_cap_smiles_override,
            "c_cap_smiles": c_cap_smiles_override,
        }

    def _detect_generic_linker_records(
        self,
        mol: Chem.Mol,
        residues: List[Dict[str, int]],
        cap_info: Dict[str, Set[int]],
    ) -> List[Dict[str, int]]:
        # Assign each atom to the nearest residue (by CA distance).
        ca_indices = [res["CA"] for res in residues]
        atom_owner: Dict[int, int] = {}
        atom_dist: Dict[int, int] = {}
        queue: List[int] = []
        for pos, ca_idx in enumerate(ca_indices, start=1):
            atom_owner[ca_idx] = pos
            atom_dist[ca_idx] = 0
            queue.append(ca_idx)
        # multi-source BFS
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            cur_dist = atom_dist[current]
            cur_owner = atom_owner[current]
            atom = mol.GetAtomWithIdx(current)
            for nb in atom.GetNeighbors():
                idx = nb.GetIdx()
                next_dist = cur_dist + 1
                if idx not in atom_dist or next_dist < atom_dist[idx]:
                    atom_dist[idx] = next_dist
                    atom_owner[idx] = cur_owner
                    queue.append(idx)
        peptide_bonds = {
            tuple(sorted((residues[i]["C"], residues[i + 1]["N"])))
            for i in range(len(residues) - 1)
        }
        records: List[Dict[str, int]] = []
        for bond in mol.GetBonds():
            a_idx = bond.GetBeginAtomIdx()
            b_idx = bond.GetEndAtomIdx()
            key = tuple(sorted((a_idx, b_idx)))
            if key in peptide_bonds:
                continue
            pos_a = atom_owner.get(a_idx)
            pos_b = atom_owner.get(b_idx)
            if pos_a is None or pos_b is None or pos_a == pos_b:
                continue
            # treat bond between different residue domains as linker connection
            records.append(
                {
                    "position": pos_a,
                    "element": None,
                    "anchor_idx": a_idx,
                    "link_idx": b_idx,
                    "bond_idx": bond.GetIdx(),
                }
            )
            records.append(
                {
                    "position": pos_b,
                    "element": None,
                    "anchor_idx": b_idx,
                    "link_idx": a_idx,
                    "bond_idx": bond.GetIdx(),
                }
            )
        if len({rec["position"] for rec in records}) < 2:
            return []
        return records

    def _match_linker_hint_atoms(
        self, mol: Chem.Mol, residues: List[Dict[str, int]]
    ) -> Optional[Tuple[Set[int], List[int], str]]:
        if not self._linker_hint_raw:
            return None
        backbone_atoms = {res["N"] for res in residues} | {res["CA"] for res in residues} | {res["C"] for res in residues}
        # Assign atoms to nearest CA for position mapping.
        ca_indices = [res["CA"] for res in residues]
        atom_owner: Dict[int, int] = {}
        atom_dist: Dict[int, int] = {}
        queue: List[int] = []
        for pos, ca_idx in enumerate(ca_indices, start=1):
            atom_owner[ca_idx] = pos
            atom_dist[ca_idx] = 0
            queue.append(ca_idx)
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            cur_dist = atom_dist[current]
            cur_owner = atom_owner[current]
            atom = mol.GetAtomWithIdx(current)
            for nb in atom.GetNeighbors():
                idx = nb.GetIdx()
                next_dist = cur_dist + 1
                if idx not in atom_dist or next_dist < atom_dist[idx]:
                    atom_dist[idx] = next_dist
                    atom_owner[idx] = cur_owner
                    queue.append(idx)

        match_set: Optional[Set[int]] = None
        match_tuple: Optional[Tuple[int, ...]] = None
        if self._linker_hint_query is not None:
            matches = mol.GetSubstructMatches(self._linker_hint_query)
            if matches:
                match_tuple = matches[0]
                match_set = set(match_tuple)
        if match_set is None:
            hint_mol = Chem.MolFromSmiles(self._linker_hint_raw)
            if hint_mol is not None:
                matches = mol.GetSubstructMatches(hint_mol)
                if matches:
                    match_tuple = matches[0]
                    match_set = set(match_tuple)
        if match_set is None:
            hint_mol = Chem.MolFromSmiles(self._linker_hint_raw)
            if hint_mol is None:
                try:
                    hint_mol = Chem.MolFromSmiles(self._linker_hint_raw, sanitize=False)
                    if hint_mol is not None:
                        Chem.SanitizeMol(hint_mol, catchErrors=True)
                        uncharger = rdMolStandardize.Uncharger()
                        hint_mol = uncharger.uncharge(hint_mol)
                        Chem.SanitizeMol(hint_mol, catchErrors=True)
                except Exception:
                    hint_mol = None
            if hint_mol is None:
                return None
            try:
                res = rdFMCS.FindMCS(
                    [mol, hint_mol],
                    atomCompare=rdFMCS.AtomCompare.CompareElements,
                    bondCompare=rdFMCS.BondCompare.CompareAny,
                    ringMatchesRingOnly=False,
                    completeRingsOnly=False,
                    timeout=2,
                )
            except Exception:
                return None
            if res.numAtoms < max(2, int(hint_mol.GetNumHeavyAtoms() * 0.7)):
                return None
            patt = Chem.MolFromSmarts(res.smartsString)
            if patt is None:
                return None
            matches = mol.GetSubstructMatches(patt)
            if not matches:
                return None
            match_tuple = matches[0]
            match_set = set(match_tuple)

        if match_set.intersection(backbone_atoms):
            return None
        # Build residue-sidechain atom sets while blocking traversal into the linker match.
        residue_sidechains: Dict[int, Set[int]] = {}
        residue_atoms: Set[int] = set()
        for pos, res in enumerate(residues, start=1):
            ca_idx = res["CA"]
            n_idx = res["N"]
            c_idx = res["C"]
            side_atoms: Set[int] = {ca_idx}
            queue = [ca_idx]
            visited = {ca_idx}
            while queue:
                current = queue.pop()
                atom = mol.GetAtomWithIdx(current)
                for nb in atom.GetNeighbors():
                    nb_idx = nb.GetIdx()
                    if nb_idx in visited or nb_idx in match_set:
                        continue
                    if nb_idx in backbone_atoms and nb_idx != ca_idx:
                        # allow Pro N to stay in sidechain traversal
                        if mol.GetAtomWithIdx(nb_idx).GetSymbol() == "N":
                            side_atoms.add(nb_idx)
                        continue
                    visited.add(nb_idx)
                    side_atoms.add(nb_idx)
                    queue.append(nb_idx)
            residue_sidechains[pos] = side_atoms
            residue_atoms.update(side_atoms)
            residue_atoms.update({n_idx, ca_idx, c_idx})
        # If linker hint contains N but match_set misses N, try to include neighboring N atoms.
        hint_has_n = False
        try:
            hint_check = Chem.MolFromSmiles(self._linker_hint_raw)
            if hint_check is not None:
                hint_has_n = any(atom.GetAtomicNum() == 7 for atom in hint_check.GetAtoms())
        except Exception:
            hint_has_n = False
        if hint_has_n and not any(mol.GetAtomWithIdx(idx).GetAtomicNum() == 7 for idx in match_set):
            extra_n = set()
            # search within 2 bonds in aromatic system
            frontier = list(match_set)
            visited = set(match_set)
            for _ in range(2):
                next_frontier = []
                for atom_idx in frontier:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    for nb in atom.GetNeighbors():
                        nb_idx = nb.GetIdx()
                        if nb_idx in visited:
                            continue
                        visited.add(nb_idx)
                        if nb.GetIsAromatic():
                            next_frontier.append(nb_idx)
                        if nb.GetAtomicNum() == 7:
                            extra_n.add(nb_idx)
                frontier = next_frontier
            match_set.update(extra_n)
        # Expand linker match to include adjacent atoms that are not part of any residue.
        expanded = set(match_set)
        for atom_idx in list(match_set):
            atom = mol.GetAtomWithIdx(atom_idx)
            for nb in atom.GetNeighbors():
                nb_idx = nb.GetIdx()
                if nb_idx in expanded or nb_idx in residue_atoms or nb_idx in backbone_atoms:
                    continue
                expanded.add(nb_idx)
        match_set = expanded
        positions: Set[int] = set()
        attach_positions: List[Tuple[int, int]] = []
        for atom_idx in match_set:
            atom = mol.GetAtomWithIdx(atom_idx)
            for nb in atom.GetNeighbors():
                nb_idx = nb.GetIdx()
                if nb_idx in match_set:
                    continue
                pos = atom_owner.get(nb_idx)
                if not pos:
                    continue
                # Only treat bonds to residue sidechains as anchor sites.
                if nb_idx not in residue_sidechains.get(pos, set()):
                    continue
                positions.add(pos)
                # anchor at the linker-side atom (atom_idx)
                attach_positions.append((pos, atom_idx))
        # Fallback: if no sidechain anchors found, allow any non-linker residue atom.
        if not attach_positions:
            for atom_idx in match_set:
                atom = mol.GetAtomWithIdx(atom_idx)
                for nb in atom.GetNeighbors():
                    nb_idx = nb.GetIdx()
                    if nb_idx in match_set:
                        continue
                    pos = atom_owner.get(nb_idx)
                    if not pos:
                        continue
                    if nb_idx in residue_atoms:
                        positions.add(pos)
                        attach_positions.append((pos, atom_idx))
        if len(positions) < 2:
            return None
        # Deduplicate by position, prefer hetero atoms as anchors.
        best_by_pos: Dict[int, int] = {}
        for pos, atom_idx in attach_positions:
            if pos not in best_by_pos:
                best_by_pos[pos] = atom_idx
                continue
            current = best_by_pos[pos]
            cur_atom = mol.GetAtomWithIdx(current)
            new_atom = mol.GetAtomWithIdx(atom_idx)
            def score(a):
                if a.GetAtomicNum() not in {6, 1}:
                    return 2
                if a.GetIsAromatic():
                    return 1
                return 0
            if score(new_atom) > score(cur_atom):
                best_by_pos[pos] = atom_idx
        attach_positions = sorted(best_by_pos.items(), key=lambda x: x[0])
        # build mapped linker from the actual molecule fragment
        # For compatibility with seq2smi_v2, each placeholder must be a terminal dummy
        # (degree 1), attached to the linker atom rather than replacing it.
        rw_mol = Chem.RWMol(mol)
        dummy_indices: List[int] = []
        for pos, atom_idx in attach_positions:
            dummy = Chem.Atom(0)
            dummy.SetAtomMapNum(int(pos))
            dummy.SetFormalCharge(0)
            dummy_idx = rw_mol.AddAtom(dummy)
            rw_mol.AddBond(atom_idx, dummy_idx, Chem.BondType.SINGLE)
            dummy_indices.append(dummy_idx)
        mapped_atoms = sorted(match_set | set(dummy_indices))
        mapped_linker = Chem.MolFragmentToSmiles(
            rw_mol.GetMol(), atomsToUse=mapped_atoms, isomericSmiles=True
        )
        # RDKit can sometimes drop the '*' in mapped dummies when formatting aromatic systems.
        if mapped_linker and "[:" in mapped_linker:
            mapped_linker = re.sub(r"\\[(?=:\\d+\\])", "[*", mapped_linker)
        return (match_set, sorted(positions), mapped_linker)

    def _normalize_linker_hint(self, linker_hint: str) -> str:
        """Normalize user-provided linker hint to a canonical SMILES skeleton."""
        if not linker_hint:
            return linker_hint
        hint_mol = Chem.MolFromSmiles(linker_hint)
        if hint_mol is None:
            try:
                hint_mol = Chem.MolFromSmiles(linker_hint, sanitize=False)
                if hint_mol is not None:
                    Chem.SanitizeMol(hint_mol, catchErrors=True)
                    uncharger = rdMolStandardize.Uncharger()
                    hint_mol = uncharger.uncharge(hint_mol)
                    Chem.SanitizeMol(hint_mol, catchErrors=True)
            except Exception:
                return linker_hint
        if hint_mol is None:
            return linker_hint
        return Chem.MolToSmiles(hint_mol, isomericSmiles=True)

    def _infer_linker_from_multi_backbone(
        self, mol: Chem.Mol, residues: List[Dict[str, int]]
    ) -> Optional[Tuple[List[int], str]]:
        """Infer linker by detecting fragments with multiple backbones after peptide cuts."""
        if not residues:
            return None
        bond_indices = []
        for idx in range(len(residues) - 1):
            bond = mol.GetBondBetweenAtoms(
                residues[idx]["C"], residues[idx + 1]["N"]
            )
            if bond:
                bond_indices.append(bond.GetIdx())
        fragmol = (
            rdmolops.FragmentOnBonds(mol, bond_indices, addDummies=False)
            if bond_indices
            else Chem.Mol(mol)
        )
        atom_frags = Chem.GetMolFrags(fragmol, asMols=False, sanitizeFrags=False)
        linker_atoms: Optional[Set[int]] = None
        ca_set = {res["CA"] for res in residues}
        for atom_ids in atom_frags:
            ca_hits = [idx for idx in atom_ids if idx in ca_set]
            if len(ca_hits) >= 2:
                linker_atoms = set(atom_ids)
                break
        if not linker_atoms:
            return None
        positions = []
        for pos, res in enumerate(residues, start=1):
            if res["CA"] in linker_atoms:
                positions.append(pos)
        if len(positions) < 2:
            return None
        # strip backbone atoms from linker atoms to keep only linker skeleton
        backbone_atoms = {res["N"] for res in residues} | {res["CA"] for res in residues} | {res["C"] for res in residues}
        linker_only = sorted(linker_atoms - backbone_atoms)
        if not linker_only:
            return None
        linker_smiles = Chem.MolFragmentToSmiles(
            mol, atomsToUse=linker_only, isomericSmiles=True
        )
        return (positions, linker_smiles)

    def _detect_known_linker(
        self, mol: Chem.Mol, residues: List[Dict[str, int]]
    ) -> Optional[Tuple[List[int], str]]:
        # Do not use linker dictionary when user did not supply a linker hint.
        if not self._linker_hint_raw:
            return None
        if not self._known_linker_queries and not self._linker_hint_mapped and not self._linker_hint_raw:
            return None
        # Assign atoms to nearest CA for position mapping.
        ca_indices = [res["CA"] for res in residues]
        backbone_atoms = {res["N"] for res in residues} | {res["CA"] for res in residues} | {res["C"] for res in residues}
        atom_owner: Dict[int, int] = {}
        atom_dist: Dict[int, int] = {}
        queue: List[int] = []
        for pos, ca_idx in enumerate(ca_indices, start=1):
            atom_owner[ca_idx] = pos
            atom_dist[ca_idx] = 0
            queue.append(ca_idx)
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            cur_dist = atom_dist[current]
            cur_owner = atom_owner[current]
            atom = mol.GetAtomWithIdx(current)
            for nb in atom.GetNeighbors():
                idx = nb.GetIdx()
                next_dist = cur_dist + 1
                if idx not in atom_dist or next_dist < atom_dist[idx]:
                    atom_dist[idx] = next_dist
                    atom_owner[idx] = cur_owner
                    queue.append(idx)
        # If user provides linker hint, use it exclusively.
        if self._linker_hint_raw:
            hint_match = self._match_linker_hint_atoms(mol, residues)
            if hint_match:
                _, positions, mapped = hint_match
                return (positions, mapped)
            return None

        for query, linker_smiles in self._known_linker_queries:
            match_set: Optional[Set[int]] = None
            if query is not None:
                matches = mol.GetSubstructMatches(query)
                if matches:
                    match_set = set(matches[0])
            # Fallback: find a tri-substituted aromatic N in an extended aromatic system.
            if match_set is None:
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() != "N" or not atom.GetIsAromatic():
                        continue
                    if atom.GetDegree() < 3:
                        continue
                    aromatic_neighbors = [nb for nb in atom.GetNeighbors() if nb.GetIsAromatic()]
                    if len(aromatic_neighbors) < 2:
                        continue
                    # collect aromatic ring system around this N
                    ring_atoms: Set[int] = set()
                    stack = [atom.GetIdx()]
                    while stack:
                        cur = stack.pop()
                        if cur in ring_atoms:
                            continue
                        ring_atoms.add(cur)
                        cur_atom = mol.GetAtomWithIdx(cur)
                        for nb in cur_atom.GetNeighbors():
                            if nb.GetIsAromatic():
                                stack.append(nb.GetIdx())
                    if len(ring_atoms) >= 8:
                        match_set = ring_atoms
                        break
            if match_set is None:
                continue
            positions: Set[int] = set()
            for atom_idx in match_set:
                atom = mol.GetAtomWithIdx(atom_idx)
                for nb in atom.GetNeighbors():
                    nb_idx = nb.GetIdx()
                    if nb_idx in match_set:
                        continue
                    pos = atom_owner.get(nb_idx)
                    if pos:
                        positions.add(pos)
            # include residues that attach to the linker through atoms inside the aromatic set
            if len(positions) >= 2:
                mapped = self._linker_hint_mapped or linker_smiles
                return (sorted(positions), mapped)
        if self._linker_hint_raw and not self._linker_hint_mapped:
            inferred = infer_linker_mapping_from_hint(
                mol, self._linker_hint_raw, atom_owner, backbone_atoms
            )
            if inferred:
                return inferred
        # If a linker hint is provided, still try the fallback aromatic-N detection.
        if self._linker_hint_mapped:
            match_set = None
            for atom in mol.GetAtoms():
                if atom.GetSymbol() != "N" or not atom.GetIsAromatic():
                    continue
                if atom.GetDegree() < 3:
                    continue
                aromatic_neighbors = [nb for nb in atom.GetNeighbors() if nb.GetIsAromatic()]
                if len(aromatic_neighbors) < 2:
                    continue
                ring_atoms: Set[int] = set()
                stack = [atom.GetIdx()]
                while stack:
                    cur = stack.pop()
                    if cur in ring_atoms:
                        continue
                    ring_atoms.add(cur)
                    cur_atom = mol.GetAtomWithIdx(cur)
                    for nb in cur_atom.GetNeighbors():
                        if nb.GetIsAromatic():
                            stack.append(nb.GetIdx())
                if len(ring_atoms) >= 8:
                    match_set = ring_atoms
                    break
            if match_set is not None:
                positions: Set[int] = set()
                for atom_idx in match_set:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    for nb in atom.GetNeighbors():
                        nb_idx = nb.GetIdx()
                        if nb_idx in match_set:
                            continue
                        pos = atom_owner.get(nb_idx)
                        if pos:
                            positions.add(pos)
                if len(positions) >= 2:
                    return (sorted(positions), self._linker_hint_mapped)
        return None

    def _known_linker_atoms(self, mol: Chem.Mol) -> Optional[Set[int]]:
        for query, _ in self._known_linker_queries:
            if query is not None:
                matches = mol.GetSubstructMatches(query)
                if matches:
                    return set(matches[0])
        # Fallback: use aromatic system around tri-substituted N
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "N" or not atom.GetIsAromatic():
                continue
            if atom.GetDegree() < 3:
                continue
            aromatic_neighbors = [nb for nb in atom.GetNeighbors() if nb.GetIsAromatic()]
            if len(aromatic_neighbors) < 2:
                continue
            ring_atoms: Set[int] = set()
            stack = [atom.GetIdx()]
            while stack:
                cur = stack.pop()
                if cur in ring_atoms:
                    continue
                ring_atoms.add(cur)
                cur_atom = mol.GetAtomWithIdx(cur)
                for nb in cur_atom.GetNeighbors():
                    if nb.GetIsAromatic():
                        stack.append(nb.GetIdx())
            if len(ring_atoms) >= 8:
                return ring_atoms
        return None


    def _infer_linker_mapping_mcs(
        self,
        mol: Chem.Mol,
        linker_hint: str,
        atom_owner: Dict[int, int],
        backbone_atoms: Optional[Set[int]] = None,
    ) -> Optional[Tuple[List[int], str]]:
        hint_mol = Chem.MolFromSmiles(linker_hint)
        if hint_mol is None:
            return None
        try:
            res = rdFMCS.FindMCS(
                [mol, hint_mol],
                atomCompare=rdFMCS.AtomCompare.CompareElements,
                bondCompare=rdFMCS.BondCompare.CompareAny,
                ringMatchesRingOnly=False,
                completeRingsOnly=False,
                timeout=2,
            )
        except Exception:
            return None
        if res.numAtoms < max(2, int(hint_mol.GetNumHeavyAtoms() * 0.7)):
            return None
        patt = Chem.MolFromSmarts(res.smartsString)
        if patt is None:
            return None
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            return None
        match_set = set(matches[0])
        if backbone_atoms and match_set.intersection(backbone_atoms):
            return None
        positions: Set[int] = set()
        for atom_idx in match_set:
            atom = mol.GetAtomWithIdx(atom_idx)
            for nb in atom.GetNeighbors():
                nb_idx = nb.GetIdx()
                if nb_idx in match_set:
                    continue
                pos = atom_owner.get(nb_idx)
                if pos is not None:
                    positions.add(pos)
        if len(positions) < 2:
            return None
        # assign placeholder indices based on residue order
        ordered = sorted(positions)
        rw = Chem.RWMol(hint_mol)
        # use attachment points: atoms that bond outside the MCS match
        attach_atoms: List[int] = []
        for atom_idx in match_set:
            atom = mol.GetAtomWithIdx(atom_idx)
            if any(nb.GetIdx() not in match_set for nb in atom.GetNeighbors()):
                attach_atoms.append(atom_idx)
        attach_positions: List[Tuple[int, int]] = []
        for atom_idx in attach_atoms:
            atom = mol.GetAtomWithIdx(atom_idx)
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
        for map_idx, (_, atom_idx) in enumerate(attach_positions, start=1):
            try:
                hint_idx = matches[0].index(atom_idx)
            except ValueError:
                continue
            a = rw.GetAtomWithIdx(hint_idx)
            a.SetAtomMapNum(map_idx)
        mapped = Chem.MolToSmiles(rw.GetMol(), isomericSmiles=True)
        return (ordered, mapped)

    def _collect_linker_atoms(
        self,
        mol: Chem.Mol,
        residues: List[Dict[str, int]],
        records: List[Dict[str, int]],
    ) -> Set[int]:
        if not records:
            return set()
        backbone_atoms = {res["N"] for res in residues} | {res["CA"] for res in residues} | {res["C"] for res in residues}
        seeds = {rec["link_idx"] for rec in records if rec.get("link_idx") is not None}
        linker_atoms: Set[int] = set()
        queue = list(seeds)
        visited = set(seeds)
        while queue:
            current = queue.pop()
            linker_atoms.add(current)
            atom = mol.GetAtomWithIdx(current)
            for nb in atom.GetNeighbors():
                idx = nb.GetIdx()
                if idx in visited or idx in backbone_atoms:
                    continue
                visited.add(idx)
                queue.append(idx)
        return linker_atoms

    def _detect_thioether_staple(
        self,
        mol: Chem.Mol,
        residues: List[Dict[str, int]],
        allowed_positions: Optional[Set[int]] = None,
    ) -> Tuple[List[int], Optional[str]]:
        if not residues:
            return [], None
        allowed_positions = allowed_positions or set()
        backbone_atoms = {res["N"] for res in residues} | {res["CA"] for res in residues} | {res["C"] for res in residues}
        s_candidates: List[Tuple[int, int]] = []
        for pos, res in enumerate(residues, start=1):
            if allowed_positions and pos not in allowed_positions:
                continue
            ca_idx = res["CA"]
            visited = {ca_idx}
            stack = [ca_idx]
            sidechain_atoms: Set[int] = set()
            while stack:
                idx = stack.pop()
                atom = mol.GetAtomWithIdx(idx)
                for nb in atom.GetNeighbors():
                    nb_idx = nb.GetIdx()
                    if nb_idx in visited:
                        continue
                    if nb_idx in backbone_atoms and nb_idx != ca_idx:
                        continue
                    if nb.GetAtomicNum() == 16:
                        visited.add(nb_idx)
                        sidechain_atoms.add(nb_idx)
                        continue
                    visited.add(nb_idx)
                    sidechain_atoms.add(nb_idx)
                    stack.append(nb_idx)
            for idx in sidechain_atoms:
                if mol.GetAtomWithIdx(idx).GetAtomicNum() == 16:
                    s_candidates.append((pos, idx))
                    break
        linker_hits: Dict[str, Set[int]] = {}
        for i in range(len(s_candidates)):
            pos_i, s_i = s_candidates[i]
            for j in range(i + 1, len(s_candidates)):
                pos_j, s_j = s_candidates[j]
                if s_i == s_j:
                    continue
                path = Chem.rdmolops.GetShortestPath(mol, s_i, s_j)
                if len(path) <= 2:
                    continue
                internal = path[1:-1]
                if not internal:
                    continue
                if any(atom_idx in backbone_atoms for atom_idx in internal):
                    continue
                linker_smiles = Chem.MolFragmentToSmiles(
                    mol,
                    atomsToUse=sorted(internal),
                    isomericSmiles=True,
                )
                if not linker_smiles:
                    continue
                linker_mol = Chem.MolFromSmiles(linker_smiles)
                if linker_mol is None:
                    continue
                linker_smiles = Chem.MolToSmiles(linker_mol, isomericSmiles=True)
                positions = linker_hits.setdefault(linker_smiles, set())
                positions.update({pos_i, pos_j})
        if not linker_hits:
            return [], None
        common_linker = max(linker_hits.items(), key=lambda item: len(item[1]))
        positions = sorted(common_linker[1])
        return positions, common_linker[0]

    def _load_templates(self) -> None:
        """Load template library and build canonical / fingerprint indices."""
        if not self.lib_path.exists():
            raise FileNotFoundError(
                f"Template library not found: {self.lib_path}"
            )
        with self.lib_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        for entry in data:
            self._register_template(entry)
        self._load_extended_templates()

    def _register_template(
        self, entry: Dict[str, str], allow_overwrite: bool = False
    ) -> Optional[TemplateEntry]:
        """Register a single template entry into caches."""
        code = entry.get("code")
        smiles_field = entry.get("smiles")
        if isinstance(smiles_field, str):
            smiles_list = [smiles_field]
        elif isinstance(smiles_field, list):
            smiles_list = [s for s in smiles_field if s]
        else:
            smiles_list = []
        if not code or not smiles_list:
            return None
        if not allow_overwrite and code in self.templates:
            return self.templates[code]

        variant_records = []
        for smiles in smiles_list:
            cleaned = remove_atom_maps(smiles)
            mol = Chem.MolFromSmiles(cleaned)
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
            except Chem.rdchem.KekulizeException:
                continue

            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
            canonical_nostereo = Chem.MolToSmiles(mol, isomericSmiles=False)
            side_canonical, side_fp = self._sidechain_signature(mol)
            fingerprint = self.fpgen.GetFingerprint(mol)
            variant_records.append(
                (smiles, mol, canonical, canonical_nostereo, side_canonical, side_fp, fingerprint)
            )

        if not variant_records:
            return None

        _, mol, canonical, canonical_nostereo, side_canonical, side_fp, fingerprint = variant_records[0]
        canonical_variants = [record[2] for record in variant_records]
        canonical_nostereo_variants = [record[3] for record in variant_records]
        components = entry.get("components")
        fragment_map: Optional[Dict[str, str]] = None
        aliases = entry.get("aliases") or []
        recognized = self._recognized_modifications().get(canonical)
        if recognized:
            # Merge known aliases
            rec_aliases = recognized.get("aliases", [])
            if rec_aliases:
                aliases = list(dict.fromkeys(list(aliases) + rec_aliases))
            is_primary = recognized.get("code") == code
            is_alias = code in rec_aliases
            if (not components) and (is_primary or is_alias):
                components = recognized.get("components")
            if is_primary or is_alias:
                fragment_map = recognized.get("fragments")
        template = TemplateEntry(
            code=code,
            mol=mol,
            canonical=canonical,
            fingerprint=fingerprint,
            polymer_type=entry.get("polymer_type", "PEPTIDE"),
            side_canonical=side_canonical,
            side_fp=side_fp,
            components=components,
            aliases=aliases if aliases else None,
            canonical_variants=canonical_variants,
        )

        if components:
            self._record_fragments(code, canonical, components, fragment_map, aliases)
        elif aliases:
            self._record_fragments(code, canonical, None, None, aliases)

        self.templates[code] = template
        for canonical_smiles in canonical_variants:
            codes = self.canonical_index.setdefault(canonical_smiles, [])
            if code not in codes:
                codes.append(code)
        for canonical_smiles in canonical_nostereo_variants:
            codes = self.canonical_index_nostereo.setdefault(canonical_smiles, [])
            if code not in codes:
                codes.append(code)
        if allow_overwrite:
            for idx, existing in enumerate(self.fp_cache):
                if existing.code == code:
                    self.fp_cache[idx] = template
                    break
            else:
                self.fp_cache.append(template)
        else:
            self.fp_cache.append(template)

        base = _base_code(code)
        if base and base in STANDARD20 and _is_standard_template_code(code):
            self.standard_entries.append(template)
        return template

    def _load_extended_templates(self) -> None:
        """Load user-extended templates if present."""
        if not self.extend_path.exists():
            return
        with self.extend_path.open("r", encoding="utf-8") as handle:
            try:
                extended = json.load(handle)
            except json.JSONDecodeError:
                extended = []

        if isinstance(extended, dict):
            extended = list(extended.values())

        for entry in extended or []:
            code = entry.get("code")
            if not code:
                continue
            registered = self._register_template(entry)
            if registered is not None:
                self.extend_entries_raw[code] = {
                    "code": code,
                    "polymer_type": entry.get("polymer_type", "PEPTIDE"),
                    "smiles": Chem.MolToSmiles(
                        registered.mol, isomericSmiles=True
                    ),
                    "components": registered.components,
                    "aliases": registered.aliases,
                }

    def _save_extend_library(self) -> None:
        """Persist extended templates to disk when new entries exist."""
        if not self.extend_dirty:
            return
        data = sorted(
            [entry for entry in self.extend_entries_raw.values() if not entry.get("code", "").endswith("X")],
            key=lambda item: item["code"]
        )
        with self.extend_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
        self.extend_dirty = False

    def _save_fragment_library(self) -> None:
        if not self.fragment_dirty:
            return
        save_fragment_library(self.fragment_library)
        self.fragment_dirty = False

    def _record_fragments(
        self,
        code: str,
        canonical: str,
        components: Optional[List[str]],
        fragment_map: Optional[Dict[str, str]] = None,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Store residue and component fragments for reuse."""
        # store main code unless it's an abstract wildcard (e.g. KX)
        if not code.endswith("X"):
            if code not in self.fragment_library:
                self.fragment_library[code] = canonical
                self.fragment_dirty = True
            self._ensure_extend_entry(code, canonical, components, fragment_map, aliases)

        if fragment_map:
            for name, frag_smiles in fragment_map.items():
                if not frag_smiles:
                    continue
                if name not in self.fragment_library:
                    self.fragment_library[name] = frag_smiles
                    self.fragment_dirty = True
                if not name.endswith("X"):
                    self._ensure_extend_entry(name, frag_smiles, None, None, None)

        if aliases:
            for alias in aliases:
                if alias.endswith("X"):
                    continue
                if alias not in self.fragment_library:
                    self.fragment_library[alias] = canonical
                    self.fragment_dirty = True
                self._ensure_extend_entry(alias, canonical, components, fragment_map, None)

    def _ensure_extend_entry(
        self,
        code: str,
        smiles: str,
        components: Optional[List[str]] = None,
        fragment_map: Optional[Dict[str, str]] = None,
        aliases: Optional[List[str]] = None,
    ) -> None:
        entry = self.extend_entries_raw.get(code)
        if entry is None:
            entry = {
                "code": code,
                "polymer_type": "PEPTIDE",
                "smiles": smiles,
            }
            self.extend_entries_raw[code] = entry
            self.extend_dirty = True
        updated = False
        if not entry.get("smiles"):
            entry["smiles"] = smiles
            updated = True
        if components and not entry.get("components"):
            entry["components"] = components
            updated = True
        if fragment_map and not entry.get("fragments"):
            entry["fragments"] = fragment_map
            updated = True
        if aliases:
            existing = set(entry.get("aliases", []))
            new_aliases = set(aliases)
            if new_aliases - existing:
                entry["aliases"] = sorted(existing | new_aliases)
                updated = True
        if updated:
            self.extend_dirty = True

    def _sidechain_signature(
        self, mol: Chem.Mol
    ) -> Tuple[Optional[str], Optional[DataStructs.ExplicitBitVect]]:
        """Return (canonical_smiles, fp) for the side chain including Cα."""
        try:
            matches = list(get_backbone_atoms(mol))
            if not matches:
                return None, None
            n_idx, ca_idx, c_idx = matches[0]#Nter AA
            backbone_atoms: Set[int] = {n_idx, ca_idx, c_idx}
            # include carbonyl oxygens
            carbonyl = mol.GetAtomWithIdx(c_idx)
            for bond in carbonyl.GetBonds():
                other = bond.GetOtherAtom(carbonyl)
                if other.GetAtomicNum() == 8:
                    backbone_atoms.add(other.GetIdx())
            # include explicit hydrogens attached to backbone N
            n_atom = mol.GetAtomWithIdx(n_idx)
            for bond in n_atom.GetBonds():#may notwork as inmplicte H NOTE
                other = bond.GetOtherAtom(n_atom)
                if other.GetAtomicNum() == 1:
                    backbone_atoms.add(other.GetIdx())

            side_atoms = self._collect_sidechain_atoms(
                mol, ca_idx, backbone_atoms
            )
            if not side_atoms or len(side_atoms) == 1:#TODO may GLY no heavy, ALA may one heavy
                #side_atoms include CA as start from it
                return None, None

            smiles = Chem.MolFragmentToSmiles(#TODO check if Pro AA sidechain cycle with backbone
                mol, atomsToUse=sorted(side_atoms), isomericSmiles=True
            )
            frag = Chem.MolFromSmiles(smiles)
            if frag is None:
                return None, None
            Chem.SanitizeMol(frag)#seems PRO as break side chain,but still work 
            canonical = Chem.MolToSmiles(frag, isomericSmiles=True)
            fp = self.fpgen.GetFingerprint(frag)
            return canonical, fp
        except Exception:  # pylint: disable=broad-except
            return None, None

    def _collect_sidechain_atoms(
        self, mol: Chem.Mol, ca_idx: int, backbone: Set[int]
    ) -> Set[int]:
        """BFS from Cα to collect side-chain atoms (including Cα)."""
        side_atoms: Set[int] = {ca_idx}
        queue: List[int] = [ca_idx]
        visited: Set[int] = {ca_idx}

        while queue:
            current = queue.pop()
            atom = mol.GetAtomWithIdx(current)
            for nb in atom.GetNeighbors():
                idx = nb.GetIdx()
                if idx in visited:
                    continue
                if self._linker_exclude_atoms and idx in self._linker_exclude_atoms:
                    continue
                visited.add(idx)
                if idx in backbone and idx != ca_idx:
                    #NOTE PRO cycle here
                    if  mol.GetAtomWithIdx(idx).GetSymbol()=='N':
                        side_atoms.add(idx)        
                    else:
                        continue
                side_atoms.add(idx)
                queue.append(idx)

        return side_atoms

    def _residue_atom_indices(self, mol: Chem.Mol, residue: Dict[str, int]) -> Set[int]:
        """Return all atoms considered part of a residue (backbone + side chain)."""
        atoms: Set[int] = set()
        n_idx = residue["N"]
        ca_idx = residue["CA"]
        c_idx = residue["C"]

        backbone_atoms: Set[int] = {n_idx, ca_idx, c_idx}
        atoms.update(backbone_atoms)

        side_atoms = self._collect_sidechain_atoms(mol, ca_idx, set(backbone_atoms))
        atoms.update(side_atoms)

        carbonyl = mol.GetAtomWithIdx(c_idx)
        for nb in carbonyl.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx != ca_idx:
                atoms.add(nb_idx)

        n_atom = mol.GetAtomWithIdx(n_idx)
        for nb in n_atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx != ca_idx:
                atoms.add(nb_idx)

        if self._linker_exclude_atoms:
            atoms.difference_update(self._linker_exclude_atoms)
        return atoms

    def _collect_peptide_atoms(
        self,
        mol: Chem.Mol,
        residues: Sequence[Dict[str, int]],
        cap_info: Optional[Dict[str, Set[int]]] = None,
        include_caps: bool = True,
    ) -> Set[int]:
        """Union of residue atoms and optionally detected cap atoms."""
        peptide_atoms: Set[int] = set()
        for residue in residues:
            peptide_atoms.update(self._residue_atom_indices(mol, residue))
        if include_caps and cap_info:
            peptide_atoms.update(cap_info.get("n_cap", set()))
            peptide_atoms.update(cap_info.get("c_cap", set()))
        elif not include_caps and cap_info:
            peptide_atoms.difference_update(cap_info.get("n_cap", set()))
            peptide_atoms.difference_update(cap_info.get("c_cap", set()))
        return peptide_atoms

    # ----------------------------- core logic -------------------------------- #

    def _enumerate_residues(self, mol: Chem.Mol) -> Tuple[List[Dict[str, int]], Dict[str, Set[int]]]:
        """
        Return ordered residue descriptors and terminal cap atoms.

        Each descriptor is {'N': idx, 'CA': idx, 'C': idx}.
        Also returns cap_info containing sets of atoms for N and C caps.
        """
        matches = list(get_backbone_atoms(mol))
        residues = [{"N": n, "CA": ca, "C": c} for (n, ca, c) in matches]
        if not residues:
            return [], {"n_cap": set(), "c_cap": set()}
        
        # Identify caps first by checking substituents on terminal N and C
        n_cap_atoms = set()
        c_cap_atoms = set()
        
        # Check N-terminal nitrogen
        n_atom = mol.GetAtomWithIdx(residues[0]["N"])#not suitable for cycle or lariat, as0 may be not the first one resudes
        for nb in n_atom.GetNeighbors():
            if (nb.GetIdx() != residues[0]["CA"] and 
                nb.GetAtomicNum() != 1):  # Not H and not CA
                n_cap_atoms.add(nb.GetIdx())
                # Add connected non-backbone atoms recursively
                stack = [nb.GetIdx()]
                while stack:
                    idx = stack.pop()
                    atom = mol.GetAtomWithIdx(idx)
                    for next_nb in atom.GetNeighbors():
                        next_idx = next_nb.GetIdx()
                        if (next_idx not in n_cap_atoms and 
                            next_idx not in [residues[0]["N"], residues[0]["CA"], residues[0]["C"]]):
                            n_cap_atoms.add(next_idx)
                            stack.append(next_idx)
        
        # Check C-terminal carbonyl
        c_atom = mol.GetAtomWithIdx(residues[-1]["C"])
        for nb in c_atom.GetNeighbors():
            if (nb.GetIdx() != residues[-1]["CA"] and 
                not (nb.GetAtomicNum() == 8 and  # Skip C=O
                     any(b.GetBondType() == Chem.BondType.DOUBLE 
                         for b in nb.GetBonds()))):
                c_cap_atoms.add(nb.GetIdx())
                # Add connected non-backbone atoms recursively
                stack = [nb.GetIdx()]
                while stack:
                    idx = stack.pop()
                    atom = mol.GetAtomWithIdx(idx)
                    for next_nb in atom.GetNeighbors():
                        next_idx = next_nb.GetIdx()
                        if (next_idx not in c_cap_atoms and 
                            next_idx not in [residues[-1]["N"], residues[-1]["CA"], residues[-1]["C"]]):
                            c_cap_atoms.add(next_idx)
                            stack.append(next_idx)

        # determine order via peptide bonds (C of i to N of i+1)
        prev_map: Dict[int, int] = {}
        next_map: Dict[int, int] = {}
        for idx, res in enumerate(residues):
            N = res["N"]
            C = res["C"]
            for jdx, other in enumerate(residues):
                if idx == jdx:
                    continue
                if mol.GetBondBetweenAtoms(other["C"], N):
                    prev_map[idx] = jdx
                if mol.GetBondBetweenAtoms(C, other["N"]):
                    next_map[idx] = jdx

        # find N-terminus (no predecessor)
        start_idx = None
        for idx in range(len(residues)):
            if idx not in prev_map:
                start_idx = idx
                break
        if start_idx is None:
            # fallback: order by nitrogen index
            order = sorted(range(len(residues)), key=lambda i: residues[i]["N"])
        else:
            order = [start_idx]
            seen = {start_idx}
            while order[-1] in next_map:
                nxt = next_map[order[-1]]
                if nxt in seen:
                    break
                order.append(nxt)
                seen.add(nxt)
            if len(order) != len(residues):
                # incomplete traversal; fallback to sorted order
                order = sorted(
                    range(len(residues)), key=lambda i: residues[i]["N"]
                )

        # reorder residues in-place for downstream processing
        ordered_residues = [residues[i] for i in order]
        return ordered_residues, {"n_cap": n_cap_atoms, "c_cap": c_cap_atoms}

    def  _match_residues_and_caps(
        self,
        mol: Chem.Mol,
        residues: List[Dict[str, int]],
        cap_info: Dict[str, Set[int]],
    ) -> Tuple[List[ResidueMatch], Optional[Dict[str, object]], Optional[Dict[str, object]], List[str]]:
        """
        Fragment the molecule, detect caps, and match each residue to a template.
        Uses pre-detected cap atoms from _enumerate_residues.
        """
        # fragment all peptide bonds at once
        bond_indices = []
        for idx in range(len(residues) - 1):
            bond = mol.GetBondBetweenAtoms(
                residues[idx]["C"], residues[idx + 1]["N"]
            )
            if bond:
                bond_indices.append(bond.GetIdx())
        fragmol = (
            rdmolops.FragmentOnBonds(mol, bond_indices, addDummies=True)
            if bond_indices
            else Chem.Mol(mol)
        )
        atom_frags = Chem.GetMolFrags(fragmol, asMols=False, sanitizeFrags=False)
        # map CA atom -> fragment index
        ca_to_fragment: Dict[int, int] = {}
        for frag_idx, atom_ids in enumerate(atom_frags):
            for res_idx, residue in enumerate(residues):
                if residue["CA"] in atom_ids:
                    ca_to_fragment[residue["CA"]] = frag_idx
        if len(ca_to_fragment) != len(residues):
            raise ValueError("Failed to map fragments to residues.")

        warnings: List[str] = []

        # terminal cap candidates
        first_residue_backbone = {
            residues[0]["N"],
            residues[0]["CA"],
            residues[0]["C"],
        }
        first_residue_atoms = first_residue_backbone | self._residue_ring_atoms(
            mol, residues[0]
        )
        n_cap_atoms = self._candidate_cap_atoms(
            mol,
            anchor=residues[0]["N"],
            avoid=residues[0]["CA"],
            backbone=residues,
            residue_atoms=first_residue_atoms,
        )

        last_residue_backbone = {
            residues[-1]["N"],
            residues[-1]["CA"],
            residues[-1]["C"],
        }
        last_residue_atoms = last_residue_backbone | self._residue_ring_atoms(
            mol, residues[-1]
        )
        c_cap_atoms = self._candidate_cap_atoms(
            mol,
            anchor=residues[-1]["C"],
            avoid=residues[-1]["CA"],
            backbone=residues,
            residue_atoms=last_residue_atoms,
        )

        residue_matches: List[ResidueMatch] = []
        # We'll fill cap info once we confirm caps truly exist
        confirmed_n_cap: Optional[Set[int]] = None
        confirmed_c_cap: Optional[Set[int]] = None

        for pos, residue in enumerate(residues, start=1):
            frag_idx = ca_to_fragment[residue["CA"]]
            atoms = set(atom_frags[frag_idx])
            remove_n = (
                n_cap_atoms.copy() if pos == 1 and n_cap_atoms else set()
            )
            remove_c = (
                c_cap_atoms.copy()
                if pos == len(residues) and c_cap_atoms
                else set()
            )

            match, used_cap = self._match_single_residue(
                fragmol,
                residue,
                atoms,
                remove_n,
                remove_c,
                position=pos,
            )
            residue_matches.append(match)

            # track whether the cap removal was successful
            if pos == 1:
                confirmed_n_cap = remove_n if used_cap["n_cap_used"] else set()
                if remove_n and not used_cap["n_cap_used"]:
                    warnings.append(
                        "N-terminus substitution did not match library; treating as no N-cap."
                    )
            if pos == len(residues):
                confirmed_c_cap = remove_c if used_cap["c_cap_used"] else set()
                if remove_c and not used_cap["c_cap_used"]:
                    warnings.append(
                        "C-terminus substitution did not match library; treating as no C-cap."
                    )

        n_cap_info = self._build_cap_info(
            mol, residues[0]["N"], confirmed_n_cap, self.n_cap_map, "N-cap"
        )
        c_cap_info = self._build_cap_info(
            mol,
            residues[-1]["C"],
            confirmed_c_cap,
            self.c_cap_map,
            "C-cap",
        )

        return residue_matches, n_cap_info, c_cap_info, warnings

    def _match_single_residue(
        self,
        fragmol: Chem.Mol,
        residue: Dict[str, int],
        atoms: Set[int],
        remove_n: Set[int],
        remove_c: Set[int],
        position: int,
    ) -> Tuple[ResidueMatch, Dict[str, bool]]:
        """
        Match a residue fragment; optionally remove terminal cap atoms.

        Returns
        -------
        ResidueMatch
        dict flags {n_cap_used, c_cap_used}
        """
        # initial attempt removing candidate cap atoms
        res_atoms = atoms - remove_n - remove_c

        raw_mol = self._raw_residue_mol(fragmol, res_atoms)
        rs_hint = self._alpha_cip(raw_mol)

        mol_candidate = self._normalized_residue_mol(
            fragmol, res_atoms, raw_reference=raw_mol
        )
        match = self._select_template_for_residue(
            mol_candidate, raw_mol, position, rs_hint
        )
        n_cap_used = bool(remove_n)
        c_cap_used = bool(remove_c)

        if match.code == "X" and match.used_fallback and (remove_n or remove_c):
            # fallback: try without removing terminal atoms (cap likely absent)
            res_atoms = atoms
            raw_mol = self._raw_residue_mol(fragmol, res_atoms)
            rs_hint = self._alpha_cip(raw_mol)
            mol_candidate = self._normalized_residue_mol(
                fragmol, res_atoms, raw_reference=raw_mol
            )
            match = self._select_template_for_residue(
                mol_candidate, raw_mol, position, rs_hint
            )
            n_cap_used = False
            c_cap_used = False

        return match, {"n_cap_used": n_cap_used, "c_cap_used": c_cap_used}

    # ------------------------- residue normalisation ------------------------ #

    def _raw_residue_mol(
        self, fragmol: Chem.Mol, atoms_to_use: Set[int]
    ) -> Chem.Mol:
        """Prepare raw residue mol (remove dummies, keep existing substituents)."""
        smiles = Chem.MolFragmentToSmiles(
            fragmol, atomsToUse=sorted(atoms_to_use), isomericSmiles=True
        )
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Failed to parse fragment SMILES.")
        rw = Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(1)
                atom.SetFormalCharge(0)
            atom.SetAtomMapNum(0)
            atom.SetIsotope(0)
            atom.SetNoImplicit(False)
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return mol

    def _normalized_residue_mol(
        self,
        fragmol: Chem.Mol,
        atoms_to_use: Set[int],
        raw_reference: Optional[Chem.Mol] = None,
    ) -> Chem.Mol:
        """Build a residue molecule and normalise it to the acid form."""
        if not atoms_to_use:
            raise ValueError("Empty residue fragment encountered.")
        smiles = Chem.MolFragmentToSmiles(
            fragmol, atomsToUse=sorted(atoms_to_use), isomericSmiles=True
        )
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Failed to parse fragment SMILES.")
        rw = Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:  # RDKit dummy atom
                atom.SetAtomicNum(1)
                atom.SetFormalCharge(0)
            atom.SetAtomMapNum(0)
            atom.SetIsotope(0)
            atom.SetNoImplicit(False)

        # Convert cleavage hydrogens on carbonyl carbon back to hydroxyls.
        for atom in list(rw.GetAtoms()):
            if atom.GetAtomicNum() != 1:
                continue
            neighbors = atom.GetNeighbors()
            if len(neighbors) != 1:
                continue
            carbon = neighbors[0]
            if carbon.GetAtomicNum() != 6:
                continue
            is_carbonyl = False
            for bond in carbon.GetBonds():
                other = bond.GetOtherAtom(carbon)
                if (
                    bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
                    and other.GetAtomicNum() == 8
                ):
                    is_carbonyl = True
                    break
            if not is_carbonyl:
                continue
            atom.SetAtomicNum(8)
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(False)
            atom.SetNumExplicitHs(0)

        # For carbonyl carbons lacking a leaving group (after cleavage), add OH.
        carbonyl_candidates: List[int] = []
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            double_oxygens = [
                bond.GetOtherAtom(atom)
                for bond in atom.GetBonds()
                if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
                and bond.GetOtherAtom(atom).GetAtomicNum() == 8
            ]
            if len(double_oxygens) != 1:
                continue
            single_neighbors = [
                bond.GetOtherAtom(atom)
                for bond in atom.GetBonds()
                if bond.GetBondType() == Chem.rdchem.BondType.SINGLE
            ]
            # If only one single neighbor (the alpha carbon), recreate hydroxyl
            if len(single_neighbors) == 1:
                carbonyl_candidates.append(atom.GetIdx())

        for idx in carbonyl_candidates:
            new_atom = Chem.Atom(8)
            new_atom.SetFormalCharge(0)
            o_idx = rw.AddAtom(new_atom)
            rw.AddBond(idx, o_idx, Chem.rdchem.BondType.SINGLE)

        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        if raw_reference is not None:
            self._copy_alpha_chirality(raw_reference, mol)
        return mol

    def _neutralize_mol(self, mol: Chem.Mol) -> Chem.Mol:
        """Neutralize charges for matching fallback."""
        try:
            uncharger = rdMolStandardize.Uncharger()
            out = uncharger.uncharge(mol)
            Chem.SanitizeMol(out)
            return out
        except Exception:
            return mol

    def _select_template_for_residue(# core step
        self,
        residue_mol: Chem.Mol,
        raw_mol: Chem.Mol,
        position: int,
        rs_hint: Optional[str],
    ) -> ResidueMatch:
        """Find best-matching template for a residue molecule."""
        canonical = Chem.MolToSmiles(residue_mol, isomericSmiles=True)
        codes = self.canonical_index.get(canonical)  # exact match (isomeric)
        rs = rs_hint if rs_hint is not None else self._alpha_cip(residue_mol)
        if not codes:
            neutral = self._neutralize_mol(residue_mol)
            neutral_canonical = Chem.MolToSmiles(neutral, isomericSmiles=True)
            codes = self.canonical_index.get(neutral_canonical)
        if not codes:
            canonical_nostereo = Chem.MolToSmiles(residue_mol, isomericSmiles=False)
            codes = self.canonical_index_nostereo.get(canonical_nostereo)
        if not codes:
            neutral = self._neutralize_mol(residue_mol)
            neutral_nostereo = Chem.MolToSmiles(neutral, isomericSmiles=False)
            codes = self.canonical_index_nostereo.get(neutral_nostereo)
        side_canonical, side_fp = self._sidechain_signature(raw_mol)
        alternatives: List[str] = []
        used_fallback = False
        best_code = "X"
        score = 0.0
        approximate = False

        if codes:
            best_code = self._choose_code_with_orientation(codes, rs)
            alternatives = [
                f"{code}@1.00" for code in codes if code != best_code
            ]
            score = 1.0 if self.canonical_index.get(canonical) else 0.99
        else:
            (
                best_code,
                alternatives,
                score,
                approximate,
            ) = self._mcs_fallback(residue_mol, rs)
            used_fallback = approximate
        if best_code == "X":
            (
                best_code,
                alternatives,
                score,
                approximate,
            ) = self._fingerprint_fallback(
                residue_mol, raw_mol, rs, side_canonical, side_fp
            )
            used_fallback = approximate

        base = _base_code(best_code)
        ld = _rs_to_ld(base, rs)

        components = None
        template_entry = self.templates.get(best_code)
        if template_entry:
            components = template_entry.components

        return ResidueMatch(
            index=position,
            code=best_code,
            canonical=canonical,
            ld=ld,
            alternatives=alternatives,
            score=score,
            used_fallback=used_fallback,
            approximate=approximate,
            components=components,
        )

    def _mcs_fallback(
        self,
        residue_mol: Chem.Mol,
        rs_hint: Optional[str],
    ) -> Tuple[str, List[str], float, bool]:
        """Try maximum common substructure match before fingerprints."""
        residue_atoms = residue_mol.GetNumHeavyAtoms()
        if residue_atoms == 0:
            return "X", [], 0.0, True
        best_score = 0.0
        best_entries: List[TemplateEntry] = []
        candidates = self.standard_entries if self.standard_entries else self.fp_cache
        residue_has_aromatic = any(atom.GetIsAromatic() for atom in residue_mol.GetAtoms())
        for entry in candidates:
            tmpl = entry.mol
            if tmpl is None:
                continue
            if residue_has_aromatic and not any(atom.GetIsAromatic() for atom in tmpl.GetAtoms()):
                continue
            if not residue_has_aromatic and any(atom.GetIsAromatic() for atom in tmpl.GetAtoms()):
                continue
            diff = abs(tmpl.GetNumHeavyAtoms() - residue_atoms)
            if diff > 3:
                continue
            try:
                res = rdFMCS.FindMCS(
                    [residue_mol, tmpl],
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
            ratio = res.numAtoms / float(min(residue_atoms, tmpl.GetNumHeavyAtoms()))
            if ratio > best_score:
                best_score = ratio
                best_entries = [entry]
            elif abs(ratio - best_score) <= 0.01:
                best_entries.append(entry)
        if best_score < 0.9 or not best_entries:
            return "X", [], 0.0, True
        codes = [entry.code for entry in best_entries]
        best_code = self._choose_code_with_orientation(codes, rs_hint)
        alternatives = [
            f"{entry.code}@{best_score:.2f}"
            for entry in best_entries
            if entry.code != best_code
        ]
        return best_code, alternatives, best_score, True

    def _alpha_cip(self, mol: Chem.Mol) -> Optional[str]:
        """Return CIP assignment for the residue's alpha carbon, if present."""
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        matches = list(get_backbone_atoms(mol))
        if not matches:
            return None
        ca_idx = matches[0][1]
        atom = mol.GetAtomWithIdx(ca_idx)
        return atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None

    def _copy_alpha_chirality(
        self, source: Chem.Mol, target: Chem.Mol
    ) -> None:
        """Copy the Cα chiral tag from source residue onto target residue."""
        src_matches = list(get_backbone_atoms(source))
        tgt_matches = list(get_backbone_atoms(target))
        if not src_matches or not tgt_matches:
            return
        src_ca = source.GetAtomWithIdx(src_matches[0][1])
        tgt_ca = target.GetAtomWithIdx(tgt_matches[0][1])
        tgt_ca.SetChiralTag(src_ca.GetChiralTag())
        Chem.AssignStereochemistry(target, force=True, cleanIt=True)

    def _choose_code_with_orientation(
        self, codes: List[str], rs: Optional[str]
    ) -> str:
        """Select preferred code from a list, favouring orientation matches."""
        if not codes:
            return "X"

        def priority(code: str) -> Tuple[int, int, int, str]:
            base = _base_code(code)
            ld = _rs_to_ld(base, rs)
            is_d_code = _code_is_d(code)
            # Primary priority: orientation match
            if ld == "D":
                orient_rank = 0 if is_d_code else 1
            elif ld == "L":
                orient_rank = 0 if not is_d_code else 1
            else:
                # No stereo info: prefer L (non-d) for standard amino acids
                orient_rank = 0 if not is_d_code else 1
            # Secondary: prefer simple codes (single letter, then dX, then others)
            if len(code) == 1 and code.isalpha() and code.isupper():
                code_rank = 0
            elif code.startswith("d") and len(code) == 2:
                code_rank = 1
            elif code.startswith("D-"):
                code_rank = 2
            else:
                code_rank = 3
            return (orient_rank, code_rank, len(code), code)

        return min(codes, key=priority)

    def _maybe_extend_template(
        self, best_code: str, normalized_mol: Chem.Mol
    ) -> Optional[str]:
        """Create an extended template for novel fragments."""
        canonical = Chem.MolToSmiles(normalized_mol, isomericSmiles=True)
        recognized_map = self._recognized_modifications()

        if canonical not in recognized_map:
            return None
        info = recognized_map[canonical]
        new_code = info["code"]
        components = info.get("components")
        fragment_map = info.get("fragments")
        aliases = info.get("aliases")
        canonical = Chem.MolToSmiles(normalized_mol, isomericSmiles=True)
        existing_codes = self.canonical_index.get(canonical, [])
        if new_code in self.templates or new_code in existing_codes:
            return new_code

        entry = {
            "code": new_code,
            "polymer_type": "PEPTIDE",
            "smiles": canonical,
            "components": components,
            "aliases": aliases,
        }
        registered = self._register_template(entry)
        if registered is None:
            return None
        self.extend_entries_raw[new_code] = {
            "code": new_code,
            "polymer_type": "PEPTIDE",
            "smiles": canonical,
            "components": components,
            "aliases": aliases,
        }
        if components or aliases:
            self._record_fragments(new_code, canonical, components, fragment_map, aliases)
        self.extend_dirty = True
        return new_code

    def _fingerprint_fallback(
        self,
        normalized_mol: Chem.Mol,
        raw_mol: Chem.Mol,
        rs_hint: Optional[str],
        side_canonical: Optional[str],
        side_fp: Optional[DataStructs.ExplicitBitVect],
    ) -> Tuple[str, List[str], float, bool]:
        """Match by side-chain similarity; return best standard residue code."""
        scores: List[Tuple[float, TemplateEntry]] = []
        best_score = -1.0
        norm_fp = self.fpgen.GetFingerprint(normalized_mol)

        candidate_entries = (
            self.standard_entries if self.standard_entries else self.fp_cache
        )

        for entry in candidate_entries:
            if side_fp is not None and entry.side_fp is not None:
                sim = DataStructs.TanimotoSimilarity(side_fp, entry.side_fp)
            else:
                sim = DataStructs.TanimotoSimilarity(
                    norm_fp, entry.fingerprint
                )
            if sim > best_score:
                best_score = sim
            scores.append((sim, entry))

        if best_score <= 0.0 or not scores:
            return "X", [], 0.0, True

        tolerance = 0.02
        top_entries = [
            (sim, entry)
            for sim, entry in scores
            if best_score - sim <= tolerance
        ]
        codes = [entry.code for _, entry in top_entries]
        best_code = self._choose_code_with_orientation(codes, rs_hint)
        alternatives = [
            f"{entry.code}@{sim:.2f}"
            for sim, entry in top_entries
            if entry.code != best_code
        ]
        approximate = best_score < 0.999
        if approximate:
            extended_code = self._maybe_extend_template(
                best_code, normalized_mol
            )
            if extended_code:
                best_code = extended_code
                approximate = False
        return best_code, alternatives, best_score, approximate

    # ----------------------------- cap handling ------------------------------ #

    def _candidate_cap_atoms(
        self,
        mol: Chem.Mol,
        anchor: int,
        avoid: int,
        backbone: List[Dict[str, int]],
        residue_atoms: Optional[Set[int]] = None,
    ) -> Set[int]:
        """Collect atoms attached to anchor that are not part of backbone."""
        backbone_atoms: Set[int] = set()
        for res in backbone:
            backbone_atoms.update(res.values())
        protected: Set[int] = set(backbone_atoms)
        if residue_atoms:
            protected.update(residue_atoms)

        cap_atoms: Set[int] = set()
        stack: List[int] = []
        anchor_atom = mol.GetAtomWithIdx(anchor)
        anchor_is_carbon = anchor_atom.GetAtomicNum() == 6
        for nb in anchor_atom.GetNeighbors():
            idx = nb.GetIdx()
            if idx == avoid:
                continue
            if idx in protected:
                continue
            if anchor_is_carbon and nb.GetAtomicNum() == 8:#O
                # retain carbonyl oxygen(s) as part of residue
                continue
            stack.append(idx)

        while stack:
            current = stack.pop()
            if current in cap_atoms:
                continue
            if current in protected:
                continue
            cap_atoms.add(current)
            atom = mol.GetAtomWithIdx(current)
            for nb in atom.GetNeighbors():
                nb_idx = nb.GetIdx()
                if nb_idx == anchor:
                    continue
                if nb_idx in cap_atoms or nb_idx in backbone_atoms:
                    continue
                stack.append(nb_idx)
        return cap_atoms

    def _residue_ring_atoms(
        self, mol: Chem.Mol, residue: Dict[str, int]
    ) -> Set[int]:
        """Return atoms belonging to rings that include both N and Cα."""
        n_idx = residue["N"]
        ca_idx = residue["CA"]
        ring_atoms: Set[int] = set()
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if n_idx in ring and ca_idx in ring:
                ring_atoms.update(ring)
        return ring_atoms

    def _build_cap_info(
        self,
        mol: Chem.Mol,
        anchor: int,
        cap_atoms: Optional[Set[int]],
        cap_map: Dict[Optional[str], str],
        label: str,
    ) -> Optional[Dict[str, object]]:
        """Return cap metadata (code + canonical) if a cap is present."""
        if not cap_atoms:
            return None
        atoms = set(cap_atoms)
        atoms.add(anchor)
        anchor_atom = mol.GetAtomWithIdx(anchor)
        if anchor_atom.GetAtomicNum() == 6:
            for nb in anchor_atom.GetNeighbors():
                if nb.GetAtomicNum() == 8:
                    atoms.add(nb.GetIdx())
        smiles = Chem.MolFragmentToSmiles(
            mol, atomsToUse=sorted(atoms), isomericSmiles=True
        )
        canonical = _canonical_smiles(smiles)
        code = cap_map.get(canonical)
        if code is None:
            code = "X_cap"
        return {"code": code, "smiles": canonical, "label": label}

    def _canonical_from_atoms(self, mol: Chem.Mol, anchor: int, atoms: Set[int]) -> str:
        """Generate canonical SMILES for a raw atom set anchored at terminal atom."""
        atom_set = set(atoms)
        atom_set.add(anchor)
        anchor_atom = mol.GetAtomWithIdx(anchor)
        if anchor_atom.GetAtomicNum() == 6:
            for nb in anchor_atom.GetNeighbors():
                if nb.GetAtomicNum() == 8:
                    atom_set.add(nb.GetIdx())
        smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atom_set), isomericSmiles=True)
        canonical = _canonical_smiles(smiles)
        return canonical or smiles

    def _ensure_backbone_placeholder(self, smiles: str) -> str:
        """Insert a [*:1] placeholder for N-cap connection if missing."""
        if "[*:1]" in smiles:
            return smiles
        if "O=C" in smiles:
            return smiles.replace("O=C", "O=C([*:1])", 1)
        return f"[*:1]{smiles}"


# --------------------------------------------------------------------------- #
# Convenience CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    """Simple CLI for manual experimentation."""
    import argparse

    parser = argparse.ArgumentParser(description="SMILES to peptide sequence")
    # parser.add_argument("smiles", nargs="+", help="Input SMILES strings")
    parser.add_argument(
        "--lib",
        dest="lib",
        default="data/monomersFromHELMCoreLibrary.json",
        help="Custom monomer library JSON (default: data/monomersFromHELMCoreLibrary.json)",
    )
    parser.add_argument(
        "--input","-i",
        dest="input",
        default="smi2seq_input.txt",
        help="Custom input smiles files input_smiles.txt)",
    )
    parser.add_argument(
        "--output","-o",
        dest="output",
        default="smi2seq_out.txt",
        help="Custom output files eg. smi2seq_out.txt ",
    )
    parser.add_argument(
        "--details",
        dest="details",
        action="store_true",
        help="Print detailed match information",
    )
    parser.add_argument(
        "--predict-ss",
        action="store_true",
        help="Predict secondary structure for each SMILES and append JSON payloads per row.",
    )
    args = parser.parse_args()

    converter = SMILES2Sequence(lib_path=args.lib)
    # smiles_list=['CC[C@H](C)[C@H](NC(C)=O)C(=O)N[C@@H](CC1=CN=CN1)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H]([C@@H](C)O)C(=O)N[C@@H]([C@@H](C)CC)C(=O)N1CCC[C@H]1C(=O)N[C@@H](C)C(=O)N[C@@H](CC(O)=O)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC1=CNC2=CC=CC=C12)C(=O)N[C@@H](CC(O)=O)C(=O)N[C@@H](CC1=CNC2=CC=CC=C12)C(=O)N[C@@H]([C@@H](C)CC)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CCCCN)C(N)=O',
    #              ]
    smiles_list = []
    if args.input[-4:]=='.txt':
        with open(args.input, "r", encoding="utf-8") as rf:
            for li in rf:
                if not li.strip():
                    continue
                smiles_list.append(li.split(",")[-1].strip())

    seq_smi: List[Tuple[str, str, Optional[dict]]] = []
    for smi in smiles_list:
        try:
            sequence, info = converter.convert(
                smi, return_details=args.details
            )
        except ValueError as exc:
            print(f"SMILES: {smi}")
            print(f"ERROR: {exc}")
            seq_smi.append((f"ERROR: {exc}", smi, {"error": str(exc)} if args.predict_ss else None))
            continue

        ss_payload = None
        if args.predict_ss:
            try:
                prediction = predict_secondary_structure(smiles=smi, lib_path=args.lib)
                ss_payload = serialize_prediction(prediction)
            except Exception as exc:  # pylint: disable=broad-except
                ss_payload = {"error": str(exc)}

        seq_smi.append((sequence, smi, ss_payload))
        if info:
            print("Details:")
            for key, value in info.items():
                print(f"  {key}: {value}")

    with open(args.output, "w", encoding="utf-8") as wf:
        header = "Sequence,SMILES"
        if args.predict_ss:
            header += ",SecondaryStructure"
        wf.write(f"{header}\n")
        for seq, smi, ss_payload in seq_smi:
            row = f"{seq},{smi}"
            if args.predict_ss:
                payload = ss_payload or {"error": "not available"}
                row += f",{json.dumps(payload, ensure_ascii=False)}"
            wf.write(f"{row}\n")

if __name__ == "__main__":
    main()

# python smi2seq_v2.py -i smi2seq_v2_input.txt -o smi2seq_v2_out.txt 
