#!/usr/bin/env python3
"""
Build residue templates from:
  - data/CycPeptMPDB_Peptide_Shape_Lariat.csv
  - data/CycPeptMPDB_Peptide_Shape_Circle.csv

For each row:
  * parse SMILES (lariat or cyclic) to obtain ordered residues (details.residues)
  * align with Sequence tokens; skip pip; record first occurrence of each code
  * helmify fragment SMILES to add [H:1]/[OH:2] anchors like corelib

Outputs:
  data/CycPeptMPDB_Peptide_Shape_Lariat_residue.json
  data/CycPeptMPDB_Peptide_Shape_Circle_residue.json
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from rdkit import Chem

from seq2smi import MonomerLib
from smi2seq import SMILES2Sequence
from smi2seq_lariat import convert as smi2seq_lariat_convert
from smi2seq_cycle import smi2seq_cycle
from utils import get_backbone_atoms

BASE = Path(__file__).resolve().parent
LARIAT_INPUT = BASE / "data" / "CycPeptMPDB_Peptide_Shape_Lariat.csv"
LARIAT_OUTPUT = BASE / "data" / "CycPeptMPDB_Peptide_Shape_Lariat_residue.json"
CYCLIC_INPUT = BASE / "data" / "CycPeptMPDB_Peptide_Shape_Circle.csv"
CYCLIC_OUTPUT = BASE / "data" / "CycPeptMPDB_Peptide_Shape_Circle_residue.json"


def build_library() -> MonomerLib:
    lib = MonomerLib(str(BASE / "data" / "monomersFromHELMCoreLibrary.json"))
    for ext in ("extend_lib.json", "extend_lib_custom.json"):
        path = BASE / ext
        if not path.exists():
            continue
        with path.open() as fh:
            data = json.load(fh)
        for entry in data:
            lib.register_entry(entry)
    return lib


def _parse_sequence_field(seq_val: Any) -> List[str]:
    if isinstance(seq_val, (list, tuple)):
        return [str(t).strip() for t in seq_val if str(t).strip()]
    seq_str = str(seq_val).strip()
    if not seq_str:
        return []
    if seq_str.startswith("[") and seq_str.endswith("]"):
        try:
            parsed = ast.literal_eval(seq_str)
            if isinstance(parsed, (list, tuple)):
                return [str(t).strip().strip("'\"") for t in parsed if str(t).strip().strip("'\"")]
        except Exception:
            pass
    seq_str = seq_str.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    sep = "." if "." in seq_str else "-"
    return [tok.strip() for tok in seq_str.split(sep) if tok.strip()]


def helmify_fragment(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if any(a.GetAtomMapNum() > 0 for a in mol.GetAtoms()):
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=False)

    matches = list(get_backbone_atoms(mol))
    if not matches:
        return None
    n_idx, _, c_idx = matches[0]
    rw = Chem.RWMol(mol)

    n_atom = rw.GetAtomWithIdx(n_idx)
    h_idx = next((nb.GetIdx() for nb in n_atom.GetNeighbors() if nb.GetAtomicNum() == 1), None)
    if h_idx is None:
        h_atom = Chem.Atom(1)
        h_atom.SetNoImplicit(True)
        h_atom.SetFormalCharge(0)
        h_atom.SetAtomMapNum(1)
        h_idx = rw.AddAtom(h_atom)
        rw.AddBond(n_idx, h_idx, Chem.BondType.SINGLE)
    else:
        h_atom = rw.GetAtomWithIdx(h_idx)
        h_atom.SetAtomMapNum(1)
        h_atom.SetNoImplicit(True)
        h_atom.SetFormalCharge(0)

    carbon = rw.GetAtomWithIdx(c_idx)
    o_idx = None
    for bond in carbon.GetBonds():
        other = bond.GetOtherAtom(carbon)
        if bond.GetBondType() == Chem.BondType.SINGLE and other.GetAtomicNum() == 8:
            o_idx = other.GetIdx()
            break
    if o_idx is None:
        o_atom = Chem.Atom(8)
        o_atom.SetFormalCharge(0)
        o_idx = rw.AddAtom(o_atom)
        rw.AddBond(c_idx, o_idx, Chem.BondType.SINGLE)
    o_atom = rw.GetAtomWithIdx(o_idx)
    o_atom.SetFormalCharge(0)
    o_atom.SetAtomMapNum(2)
    o_atom.SetNoImplicit(True)
    o_atom.SetNumExplicitHs(1)

    helm_mol = rw.GetMol()
    try:
        Chem.SanitizeMol(helm_mol)
    except Exception:
        return None
    return Chem.MolToSmiles(helm_mol, isomericSmiles=True, canonical=False)


def process_dataset(path: Path, output: Path, parse_fn, lib_label: str) -> None:
    if not path.exists():
        print(f"Skip missing {path}")
        return
    df = pd.read_csv(path)
    if "SMILES" not in df.columns or "Sequence" not in df.columns:
        print(f"Skip {path}, missing required columns.")
        return
    residues: Dict[str, dict] = {}
    pre_codes = set()
    for seq_val in df["Sequence"]:
        for tok in _parse_sequence_field(seq_val):
            if tok and tok != "pip":
                pre_codes.add(tok)
    seen_codes = set()
    for _, row in df.iterrows():
        if pre_codes and seen_codes.issuperset(pre_codes):
            break
        seq_tokens = _parse_sequence_field(row["Sequence"])
        smiles = str(row["SMILES"]).strip()
        if not smiles or not seq_tokens:
            continue
        try:
            res_info = parse_fn(smiles)
        except Exception:
            continue
        if not res_info:
            continue
        span = min(len(seq_tokens), len(res_info))
        for token, info in zip(seq_tokens[:span], res_info[:span]):
            if not token or token == "pip" or token in seen_codes:
                continue
            frag = info.get("canonical") or info.get("canonical_smiles")
            if not frag:
                continue
            helm = helmify_fragment(frag) or frag
            residues[token] = {
                "code": token,
                "smiles": helm,
                "polymer_type": "PEPTIDE",
                "type": "PEPTIDE",
            }
            seen_codes.add(token)
    residue_list = list(residues.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(residue_list, fh, ensure_ascii=False, indent=2)
    print(f"[{lib_label}] Wrote {output} with {len(residue_list)} unique residues")


def main() -> None:
    lib = build_library()

    def parse_lariat(smiles: str):
        _, details = smi2seq_lariat_convert(smiles, lib)
        if isinstance(details, dict):
            return details.get("residues")
        return None

    process_dataset(LARIAT_INPUT, LARIAT_OUTPUT, parse_lariat, "lariat")

    cyc_converter = SMILES2Sequence(lib_path=str(BASE / "data" / "monomersFromHELMCoreLibrary.json"))

    def parse_cycle(smiles: str):
        _, details = smi2seq_cycle(smiles, converter=cyc_converter)
        if isinstance(details, dict):
            return details.get("residues")
        return None

    process_dataset(CYCLIC_INPUT, CYCLIC_OUTPUT, parse_cycle, "cycle")


if __name__ == "__main__":
    main()
