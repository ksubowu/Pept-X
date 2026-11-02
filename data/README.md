# Benchmark input data

The following original CycPeptMPDB CSVs are included as **benchmark/test inputs** for peptide-expression-to-SMILES and SMILES-to-expression conversion. They are byte-for-byte copies of the local source data used for manuscript evaluation. All records, including duplicate SMILES, are retained.

| File | Records | Distinct raw SMILES | SHA-256 |
|---|---:|---:|---|
| [CycPeptMPDB_Peptide_Shape_Circle.csv](CycPeptMPDB_Peptide_Shape_Circle.csv) | 5530 | 5384 | `450f0206200911b247b11a0aeb867f2fe0b62f11acfae7ec54034df1c09bed42` |
| [CycPeptMPDB_Peptide_Shape_Lariat.csv](CycPeptMPDB_Peptide_Shape_Lariat.csv) | 2936 | 2635 | `2e040240ed02e368b58b9c3e0441e4162fec9515296372bae021bb3164403e4c` |

`SMILES` is the molecular-graph reference and the input to reverse prediction. `Sequence` and `HELM` supply reference residue/connection annotations for scoring and separate forward assembly; they are not inputs to the reverse predictor.

## Fixed 18-case lariat benchmarks

| File | Records | Reference topology counts | Origin |
|---|---:|---|---|
| [CycPeptMPDB_Lariat_Manuscript_Subset.csv](CycPeptMPDB_Lariat_Manuscript_Subset.csv) | 18 | Type 1: 8; type 2: 2; type 3: 8 | The unchanged database subset displayed in the manuscript; selected from the full Lariat CSV |
| [Synthetic_Lariat_Benchmark.csv](Synthetic_Lariat_Benchmark.csv) | 18 | Six per lariat type | Independently constructed RDKit reference graphs; synthetic regression cases |

The database subset retains every original CSV column for its selected IDs and adds `PeptX_Expression`, `Topology_Class`, `Expected_Attachment` and `Reference_Source` from the frozen manuscript references. It is contained in the full 2936-row Lariat dataset and must not be counted as 18 additional independent database records.

The synthetic CSV preserves each frozen case ID, exact reference SMILES, reference expression, residue tokens, topology, attachment positions and source description. Its `Sequence` field is a JSON array of tokens; `Expected_Attachment` in both new files is a JSON array of one-based residue positions excluding caps. Synthetic structures are constructive references, rather than experimental database observations. Neither new file contains predictions used to define the expected answer.

The manuscript Circle evaluation uses a frozen 597-entry registry from its historical reproduction archive. The current `monomersFromHELMCoreLibrary.json` and root extension/custom libraries define the separately evaluated Lariat configuration. The two configurations and complete reverse-semantic criteria are documented in the root [README](../README.md#manuscript-validation-scope).

The CSV row-level `Version` fields are preserved. A historical global database release identifier or download date is not inferred from file timestamps. The Lariat dataset was used during development and regression testing; its 18-case displayed subset and 32 repaired failures are included in the full 2936-row denominator.

`monomersFromHELMCoreLibrary.json` and `linker_dict.json` in this directory are runtime conversion libraries, separate from the benchmark CSVs.
