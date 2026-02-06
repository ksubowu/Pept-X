#!/usr/bin/env python3
"""Merge template libraries; de-dup when both code and smiles match exactly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

BASE = Path(__file__).resolve().parent
INPUT_FILES = [
    BASE / "extend_lib_custom.json",
    BASE / "extend_lib.json",
    BASE / "data" / "monomersFromHELMCoreLibrary.json",
    BASE / "data" / "CycPeptMPDB_Peptide_Shape_Lariat_residue.json",
    BASE / "data" / "CycPeptMPDB_Peptide_Shape_Circle_residue.json",
]
OUTPUT_FILE = BASE / "data" / "monomersFromHELMCoreLibrary.json"


def load_entries(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = list(data.values())
    return data if isinstance(data, list) else []


def extract_code_smiles(entry: dict) -> Tuple[str, Optional[str]]:
    code = (entry.get("code") or "").strip()
    smi = entry.get("smiles") or entry.get("smiles_L") or entry.get("smiles_D")
    if isinstance(smi, list):
        smi = next((s for s in smi if isinstance(s, str) and s), None)
    return code, smi


def merge() -> List[dict]:
    seen = set()  # (code, smiles)
    merged: List[dict] = []
    for path in INPUT_FILES:
        for entry in load_entries(path):
            code, smi = extract_code_smiles(entry)
            if not code or not smi:
                continue
            key = (code, smi)
            if key in seen:
                continue
            seen.add(key)
            normalized = dict(entry)
            normalized["code"] = code
            normalized["smiles"] = smi
            merged.append(normalized)
    return merged


def main() -> None:
    merged = merge()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        json.dump(merged, fh, ensure_ascii=False, indent=2)
    print(f"Merged {len(INPUT_FILES)} files -> {len(merged)} unique (code,smiles) entries")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
