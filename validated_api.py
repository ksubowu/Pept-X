"""Graph reconstruction validation for the peptide-expression converters.

A candidate expression is not an accepted conversion until its rebuilt graph
passes the declared parent-structure oracle. Warnings are independent metadata.
"""
from molecular_graph import canonical_standardized
from seq2smi_v2 import seq2smi


def compile_expression(expression, library):
    """Route topology metadata without changing or repairing residue tokens."""
    parts=expression.split('|')
    if any(p.lower().startswith('lariat') for p in parts[1:]):
        from seq2smi_lariat import build_smiles
        return build_smiles(expression, monomer_lib=library)
    head2tail=any(p.lower()=='head2tail' for p in parts[1:])
    core=parts[0]
    metadata=[]
    for p in parts[1:]:
        if p.lower()=='head2tail':continue
        if p.lower().startswith('conn:'):
            core += '$'+p.split(':',1)[1]+'$$$'
        else:metadata.append(p)
    if head2tail and not core.endswith('[Cyclo]'):core += '.[Cyclo]'
    expression=core+''.join('|'+p for p in metadata)
    return seq2smi(expression,library)


def validate_candidate(input_smiles, sequence, details, rebuild):
    details = details or {}
    out = dict(sequence_candidate=sequence, accepted_expression=None,
               graph_equivalent=False, rebuilt_smiles=None,
               warnings=details.get('warnings', []), details=details)
    if not sequence or not sequence.strip() or sequence.startswith('ERROR:'):
        out['status'] = 'reconstruction_failure'
        return out
    try:
        smi = rebuild(sequence)
        out['rebuilt_smiles'] = smi
        ref, pred = canonical_standardized(input_smiles), canonical_standardized(smi)
        out['graph_equivalent'] = bool(ref and pred and ref == pred)
    except Exception as exc:
        out['status'] = 'unsupported_chemistry' if 'X' in sequence else 'reconstruction_failure'
        out['error'] = f'{type(exc).__name__}: {exc}'
        return out
    if not out['graph_equivalent']:
        out['status'] = 'graph_mismatch'
        return out
    out['accepted_expression'] = sequence
    residues = details.get('residues', [])
    if any(r.get('alternatives') for r in residues):
        out['status'] = 'ambiguous_decomposition'
    elif out['warnings'] or any(r.get('used_fallback') or float(r.get('score', 1)) < 1 for r in residues):
        out['status'] = 'rescued_success'
    else:
        out['status'] = 'canonical_success'
    return out


def convert_validated(smiles, converter, library, linker_hint=None):
    try:
        seq, details = converter.convert(smiles, return_details=True, linker_hint=linker_hint)
    except Exception as exc:
        return dict(status='unsupported_chemistry', sequence_candidate=None,
                    accepted_expression=None, graph_equivalent=False,
                    error=f'{type(exc).__name__}: {exc}')
    # The legacy compiler can return ring semantics only in the details object.
    # Serialize those semantics explicitly before attempting reconstruction.
    if (details or {}).get('cycle_meta',{}):
        meta=details['cycle_meta']
        if meta.get('annotation')=='head2tail' and '|head2tail' not in seq and '[Cyclo]' not in seq:
            seq += '|head2tail'
    return validate_candidate(smiles, seq, details, lambda s: compile_expression(s, library))


def main():
    import argparse, json
    from smi2seq_v2 import SMILES2Sequence
    from seq2smi_v2 import MonomerLib
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('smiles');p.add_argument('--lib',default='data/monomersFromHELMCoreLibrary.json')
    p.add_argument('--auto-linker',action='store_true');p.add_argument('--linker-hint')
    p.add_argument('--lariat',action='store_true');a=p.parse_args()
    library=MonomerLib(a.lib)
    if a.lariat:
        from monomer_library import build_library
        library=build_library(a.lib)
        from smi2seq_lariat import convert
        from seq2smi_lariat import build_smiles
        try:
            seq,details=convert(a.smiles,lib=library)
            result=validate_candidate(a.smiles,seq,details,lambda s:build_smiles(s,monomer_lib=library))
        except Exception as e:
            result=dict(status='unsupported_chemistry',accepted_expression=None,graph_equivalent=False,error=str(e))
    else:
        converter=SMILES2Sequence(lib_path=a.lib,auto_linker_detection=a.auto_linker)
        result=convert_validated(a.smiles,converter,library,a.linker_hint)
    print(json.dumps(result,indent=2,default=str))

if __name__=='__main__':main()
