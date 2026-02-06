#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for fragment library management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

FRAG_LIB_PATH = Path("frag_smi.json")


def load_fragment_library() -> Dict[str, str]:
    if FRAG_LIB_PATH.exists():
        with FRAG_LIB_PATH.open("r", encoding="utf-8") as handle:
            try:
                return json.load(handle)
            except json.JSONDecodeError:
                return {}
    return {}


def save_fragment_library(library: Dict[str, str]) -> None:
    with FRAG_LIB_PATH.open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(library.items())), handle, indent=2, ensure_ascii=False)

