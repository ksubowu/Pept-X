# Pept-x

A bidirectional converter between peptide sequences and SMILES representations, specifically designed for peptide structure analysis and generation.

## Features

- Convert peptide sequences to SMILES notation
- Convert SMILES back to peptide sequences
- Handle non-standard amino acids
- Support D-amino acids
- Support terminal modifications (N-caps and C-caps)
- Built on RDKit for robust chemical structure handling

## Installation

```bash
# Clone the repository
git clone https://github.com/ksubowu/Pept-x.git
cd Pept-x

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line Interface

1. Sequence to SMILES conversion:
```bash
python seq2smi.py -i input_sequences.txt -o output_smiles.txt -l data/monomersFromHELMCoreLibrary.json
```

2. SMILES to sequence conversion:
```bash
python smi2seq.py -i input_smiles.txt -o output_sequences.txt -l data/monomersFromHELMCoreLibrary.json
```

### Python API

```python
from seq2smi import Sequence2SMILES
from smi2seq import SMILES2Sequence

# Initialize converters
seq2smi = Sequence2SMILES('data/monomersFromHELMCoreLibrary.json')
smi2seq = SMILES2Sequence('data/monomersFromHELMCoreLibrary.json')

# Convert sequence to SMILES
sequence = "A-G-T"
smiles = seq2smi.convert(sequence)

# Convert SMILES back to sequence
sequence = smi2seq.convert(smiles)
```

## Data Format

### Input Sequence Format
- Single letter amino acid codes
- D-amino acids prefixed with 'd' or 'D-'
- Terminal modifications supported
- Fragments separated by hyphens

Example:
```
A-G-T
ac-A-dV-T-am
```

### Input SMILES Format
- Standard SMILES notation
- Atom mapping optional
- Proper peptide backbone required
- Terminal groups supported

### Library Format (JSON)
```json
[
    {
        "code": "A",
        "smiles": "CC(N)C(=O)O",
        "type": "PEPTIDE"
    },
    {
        "code": "acetyl",
        "smiles": "CC(=O)N",
        "type": "CAP_N"
    }
]
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{Pept-x2026,
  author = {Bowen Tang, et al},
  title = {Pept-X: Chemically Lossless and Reversible Translation between Peptide Sequence Expressions and Molecular Graphs},
  year = {2026},
  url = {https://github.com/ksubowu/Pept-x}
}
```
