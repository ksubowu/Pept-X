#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a PDF catalog of template molecules."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import Draw

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CORE_LIB = DATA_DIR / "monomersFromHELMCoreLibrary.json"
EXTEND_LIB = BASE_DIR / "extend_lib.json"
EXTEND_LIB_CUSTOM = BASE_DIR / "extend_lib_custom.json"
OUTPUT_PDF = DATA_DIR / "templates_lib.pdf"
MOLS_PER_ROW = 6
CELL_SIZE = (250, 250)


def _load_entries(paths: List[Path]) -> List[dict]:
    entries: List[dict] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data = list(data.values())
        if isinstance(data, list):
            entries.extend(data)
    return entries


def _normalize_smiles(smiles: str) -> Chem.Mol | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def _clean_label(label: str) -> str:
    # Remove atom-map style suffix (e.g., "[OH:2]" -> "[OH]")
    return re.sub(r":\d+\]", "]", label)


def _summarize_polymer_type(code: str, entry: dict) -> str:
    polymer = (entry.get("polymer_type") or "").upper()
    if not polymer:
        lower = code.lower()
        if lower in {"ac", "formyl"}:
            polymer = "CAP_N"
        elif lower in {"am", "ome"}:
            polymer = "CAP_C"
    return polymer or "PEPTIDE"


def _build_dataset() -> Tuple[List[Chem.Mol], List[str], Dict[str, int]]:
    entries = _load_entries([CORE_LIB, EXTEND_LIB, EXTEND_LIB_CUSTOM])
    seen_codes: set[str] = set()
    mols: List[Chem.Mol] = []
    labels: List[str] = []
    counts: Dict[str, int] = {}
    for entry in entries:
        code = entry.get("code") or entry.get("name")
        if not code or code in seen_codes:
            continue
        smiles = entry.get("smiles") or entry.get("smiles_L") or entry.get("smiles_D")
        if isinstance(smiles, list):
            smiles = next((s for s in smiles if isinstance(s, str) and s), None)
        if not isinstance(smiles, str):
            continue
        mol = _normalize_smiles(smiles)
        if mol is None:
            continue
        seen_codes.add(code)
        mols.append(mol)
        label = _clean_label(str(code))
        labels.append(label)
        key = _summarize_polymer_type(label, entry)
        counts[key] = counts.get(key, 0) + 1
    return mols, labels, counts


def _render_footer(lines: List[str], width: int) -> Image.Image:
    padding = 10
    line_height = 20
    footer_height = padding * 2 + line_height * len(lines)
    img = Image.new("RGB", (width, footer_height), "white")
    draw = ImageDraw.Draw(img)
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill="black")
        y += line_height
    return img


def build_pdf() -> Path:
    mols, labels, counts = _build_dataset()
    if not mols:
        raise RuntimeError("No molecules found across libraries.")
    grid = Draw.MolsToGridImage(
        mols,
        legends=labels,
        molsPerRow=MOLS_PER_ROW,
        subImgSize=CELL_SIZE,
    )
    width, height = grid.size
    summary_lines = [f"Total templates: {len(mols)}"]
    for key in sorted(counts):
        summary_lines.append(f"{key}: {counts[key]}")
    footer = _render_footer(summary_lines, width)
    combined = Image.new("RGB", (width, height + footer.height), "white")
    combined.paste(grid, (0, 0))
    combined.paste(footer, (0, height))
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    combined.save(OUTPUT_PDF)
    return OUTPUT_PDF


def main() -> None:
    path = build_pdf()
    print(f"Template catalog saved to {path}")


if __name__ == "__main__":
    main()
