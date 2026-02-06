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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdmolops
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from conformer import predict_secondary_structure, serialize_prediction

from utils import (
    clean_smiles,
    get_backbone_atoms,
    remove_atom_maps,
    N_CAPS,
    C_CAPS,
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
        self.fp_cache: List[TemplateEntry] = []
        self.standard_entries: List[TemplateEntry] = []
        self.extend_path = Path("extend_lib.json")
        self.extend_entries_raw: Dict[str, Dict[str, str]] = {}
        self.extend_dirty = False
        self.fragment_library: Dict[str, str] = load_fragment_library()
        self.fragment_dirty = False
        self.n_cap_map = {
            _canonical_smiles(sm): code for code, sm in N_CAPS.items()
        }
        self.c_cap_map = {
            _canonical_smiles(sm): code for code, sm in C_CAPS.items()
        }
        self._load_templates()

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
        self, smiles: str, return_details: bool = False
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
        mol = Chem.MolFromSmiles(clean)
        if mol is None:
            raise ValueError("Failed to parse SMILES.")
        Chem.SanitizeMol(mol)

        residues, cap_info = self._enumerate_residues(mol)
        if not residues:
            raise ValueError("No peptide backbone detected.")

        (
            residue_matches,
            n_cap_info,
            c_cap_info,
            warnings,
        ) = self._match_residues_and_caps(mol, residues, cap_info)

        tokens: List[str] = []
        
        # Add N-cap if detected
        if n_cap_info and n_cap_info["code"] and n_cap_info["code"] != "H":
            if n_cap_info["code"].startswith("X_cap"):
                tokens.append(f"[N-cap:{n_cap_info['smiles']}]")
            else:
                tokens.append(n_cap_info["code"])
        
        # Add residues
        for match in residue_matches:
            if match.used_fallback:
                if match.code.startswith("X-"):
                    tokens.append(f"[{match.code}]")
                else:
                    tokens.append(f"{match.code}X")
            else:
                tokens.append(match.code)
        
        # Add C-cap if detected
        if c_cap_info and c_cap_info["code"] and c_cap_info["code"] != "H":
            if c_cap_info["code"].startswith("X_cap"):
                tokens.append(f"[C-cap:{c_cap_info['smiles']}]")
            else:
                tokens.append(c_cap_info["code"])

        sequence = ".".join(tokens)
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
            side_canonical, side_fp = self._sidechain_signature(mol)
            fingerprint = self.fpgen.GetFingerprint(mol)
            variant_records.append(
                (smiles, mol, canonical, side_canonical, side_fp, fingerprint)
            )

        if not variant_records:
            return None

        primary_smiles, mol, canonical, side_canonical, side_fp, fingerprint = variant_records[0]
        canonical_variants = [record[2] for record in variant_records]
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
        n_atom = mol.GetAtomWithIdx(residues[0]["N"])
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

    def _select_template_for_residue(# core step
        self,
        residue_mol: Chem.Mol,
        raw_mol: Chem.Mol,
        position: int,
        rs_hint: Optional[str],
    ) -> ResidueMatch:
        """Find best-matching template for a residue molecule."""
        canonical = Chem.MolToSmiles(residue_mol, isomericSmiles=True)
        codes = self.canonical_index.get(canonical)#smi:AAsymbol
        rs = rs_hint if rs_hint is not None else self._alpha_cip(residue_mol)
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
            score = 1.0
        else:
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
                orient_rank = 1
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
        "--input",
        dest="input",
        default="smi2seq_input.txt",
        help="Custom input smiles files input_smiles.txt)",
    )
    parser.add_argument(
        "--output",
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
