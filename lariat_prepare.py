#!/usr/bin/env python3
"""
step 1 helper:
    read data/CycPeptMPDB_Peptide_Shape_Lariat.csv
    run lariat parsing on SMILES column
    insert columns [myseq, topology] before SMILES
    write myLariat.csv
Uses the existing lariat converter (smi2seq_lariat.convert) and the merged monomer
libraries (core + extend_lib + extend_lib_custom).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from seq2smi import MonomerLib
from smi2seq_lariat import convert as smi2seq_lariat_convert

BASE = Path(__file__).resolve().parent
INPUT = BASE / "data" / "CycPeptMPDB_Peptide_Shape_Lariat.csv"
OUTPUT = BASE / "myLariat.csv"


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


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    df = pd.read_csv(INPUT)
    if "SMILES" not in df.columns:
        raise ValueError("Input CSV must contain a 'SMILES' column.")

    lib = build_library()

    myseq: List[str] = []
    topo: List[str] = []

    for _, row in df.iterrows():
        smiles = str(row["SMILES"]).strip()
        seq_tokens = []
        if "Sequence" in row and isinstance(row["Sequence"], str):
            seq_str = str(row["Sequence"]).replace(" ", "")
            sep = "." if "." in seq_str else "-"
            seq_tokens = [t for t in seq_str.split(sep) if t]
        try:
            seq_out, details = smi2seq_lariat_convert(smiles, lib)
            core = seq_out.split("|")[0]
            sep = "." if "." in core else "-"
            tokens = core.split(sep)
            meta = ""
            if "|" in seq_out:
                meta = "|" + "|".join(seq_out.split("|")[1:])
            unmatched = []
            if isinstance(details, dict):
                unmatched = details.get("unmatched") or []
            if unmatched and seq_tokens:
                for u in unmatched:
                    idx_res = u.get("residue_index")
                    if not idx_res or idx_res < 1 or idx_res > len(seq_tokens):
                        continue
                    tokens[idx_res - 1] = seq_tokens[idx_res - 1]
            seq_final = ".".join(tokens) + (meta if meta else "|lariat")
            myseq.append(seq_final)
            topo.append("lariat")
        except Exception as exc:  # pylint: disable=broad-except
            myseq.append(f"ERROR: {exc}")
            topo.append(None)

    smi_idx = df.columns.get_loc("SMILES")
    df.insert(smi_idx, "myseq", myseq)
    df.insert(smi_idx, "topology", topo)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved parsed CSV to {OUTPUT}")


if __name__ == "__main__":
    main()
