"""Load the registered monomer inputs used by the dedicated lariat converters."""
import json
from pathlib import Path
from seq2smi import MonomerLib

ROOT = Path(__file__).resolve().parent


def build_library(core_path=None, *, include_extensions=True):
    """Load core, extension, then custom entries, preserving registered aliases.

    This is the same merge order used for the manuscript's full Lariat run.
    Loading does not register new chemistry or write any library files.
    """
    library = MonomerLib(str(core_path or ROOT/'data/monomersFromHELMCoreLibrary.json'))
    if include_extensions:
        for name in ('extend_lib.json', 'extend_lib_custom.json'):
            path = ROOT/name
            if path.exists():
                for entry in json.loads(path.read_text()):
                    library.register_entry(entry)
    return library
