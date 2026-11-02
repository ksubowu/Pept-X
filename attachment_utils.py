"""Explicit attachment selection and exact mapped-linker query matching."""
from collections import deque
import re
from rdkit import Chem


class AttachmentSelectionWarning(UserWarning):
    """A default attachment requires user review."""


def select_sidechain_anchor(fragment, preference, residue, position, backbone_finder):
    """Return (atom index, diagnostic); all indices are local to the template."""
    matches = list(backbone_finder(fragment))
    if not matches:
        raise ValueError(f"Residue {position} ({residue}): no recognized backbone.")
    n, ca, c = matches[0]
    backbone = {n, ca, c}
    backbone.update(a.GetIdx() for a in fragment.GetAtomWithIdx(c).GetNeighbors()
                    if a.GetAtomicNum() == 8)
    backbone.update(a.GetIdx() for a in fragment.GetAtoms() if a.GetAtomMapNum() in (1, 2))
    side = set()
    queue = [ca]
    seen = {ca}
    while queue:
        idx = queue.pop()
        for atom in fragment.GetAtomWithIdx(idx).GetNeighbors():
            other = atom.GetIdx()
            if other in seen or other in backbone:
                continue
            seen.add(other);side.add(other);queue.append(other)
    heavy = sorted(i for i in side if fragment.GetAtomWithIdx(i).GetAtomicNum() > 1)
    pref = (preference or '').strip().upper()
    record = dict(residue_position=position, residue_code=residue,
                  requested_atom=pref or None, warnings=[], warning_records=[])

    def warn(code, message):
        record['warnings'].append(message)
        record['warning_records'].append(dict(code=code, residue_position=position,
                                              residue_code=residue, message=message))

    # R3 and higher identify a mapped leaving group, not a backbone R1/R2 site.
    map_selector = re.fullmatch(r'R(\d+)', pref)
    leaving = []
    if map_selector:
        number = int(map_selector[1])
        if number < 3:
            raise ValueError('Side-chain selectors must use R3 or higher, not backbone R1/R2.')
        atoms = [a for a in fragment.GetAtoms() if a.GetAtomMapNum() == number]
        if len(atoms) != 1:
            raise ValueError(f'Residue {position} ({residue}): selector {pref} must identify one mapped leaving group.')
        neighbors = [a for a in atoms[0].GetNeighbors() if a.GetAtomicNum() > 1]
        if len(neighbors) != 1 or neighbors[0].GetIdx() not in heavy:
            raise ValueError(f'Residue {position} ({residue}): {pref} is not an unambiguous side-chain attachment.')
        candidates = [neighbors[0].GetIdx()]
        leaving = [atoms[0].GetIdx()]
        rule = 'explicit_template_anchor'
    else:
        candidates = heavy
        if pref:
            filtered = [i for i in heavy if fragment.GetAtomWithIdx(i).GetSymbol().upper() == pref]
            if filtered:
                candidates = filtered
            else:
                warn('requested_element_absent',
                     f'Residue {position} ({residue}) has no side-chain {pref}; a default attachment is used.')
        # Adding a neutral bond is supported here by replacing a hydrogen.
        candidates = [i for i in candidates if fragment.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True) > 0]
        if not candidates:
            raise ValueError(f'Residue {position} ({residue}): no compatible side-chain atom can replace a hydrogen; specify a mapped leaving-group selector such as R3.')
        terminal = [i for i in candidates if sum(a.GetAtomicNum() > 1 for a in fragment.GetAtomWithIdx(i).GetNeighbors()) == 1]
        if terminal:
            candidates = terminal
            rule = 'terminal_sidechain_heavy_atom'
        else:
            rule = 'nonterminal_sidechain_fallback'
            warn('no_terminal_sidechain_atom',
                 f'Residue {position} ({residue}) has no compatible terminal side-chain heavy atom; a nonterminal atom is selected. Specify an explicit template anchor to remove this uncertainty.')
        if len(candidates) > 1:
            warn('ambiguous_sidechain_attachment',
                 f'Residue {position} ({residue}) has {len(candidates)} compatible side-chain attachment atoms; a deterministic default is selected. Specify an element or an explicit template anchor such as R3.')
    distances = {i: len(Chem.GetShortestPath(fragment, ca, i)) - 1 for i in candidates}
    chosen = min(candidates, key=lambda i: (-distances[i], i))
    record.update(selection_rule=rule, selected_template_atom_index=chosen,
                  selected_element=fragment.GetAtomWithIdx(chosen).GetSymbol(),
                  candidate_template_atom_indices=candidates,
                  leaving_template_atom_indices=leaving)
    return chosen, record


def mapped_linker_match(mol, residues, hint_smiles):
    """Match complete linker boundaries using supplied residue-position maps.

    Actual attachment atoms come from the target graph, including internal
    side-chain atoms. They are never replaced by a terminal-atom heuristic.
    """
    hint = Chem.MolFromSmiles(hint_smiles)
    if hint is None:
        raise ValueError('Cannot parse mapped linker hint.')
    dummies = [a for a in hint.GetAtoms() if a.GetAtomicNum() == 0]
    labels = [a.GetAtomMapNum() for a in dummies]
    if len(dummies) < 2 or any(n <= 0 for n in labels) or len(set(labels)) != len(labels):
        raise ValueError('Mapped linker hints require at least two uniquely numbered terminal [*:residue_position] sites.')
    if any(a.GetDegree() != 1 for a in dummies):
        raise ValueError('Each mapped linker placeholder must have exactly one neighbor.')
    if any(n > len(residues) for n in labels):
        raise ValueError('Mapped linker hint refers to a residue position outside the detected sequence.')
    query = Chem.RWMol(hint)
    for atom in dummies:
        wildcard = Chem.AtomFromSmarts('*')
        wildcard.SetAtomMapNum(atom.GetAtomMapNum())
        query.ReplaceAtom(atom.GetIdx(), wildcard)
    core_ids = [a.GetIdx() for a in hint.GetAtoms() if a.GetAtomicNum() != 0]
    if not core_ids:
        raise ValueError('Mapped linker hint contains no linker-core atoms.')
    backbone = {r[k] for r in residues for k in ('N', 'CA', 'C')}
    for r in residues:
        backbone.update(a.GetIdx() for a in mol.GetAtomWithIdx(r['C']).GetNeighbors() if a.GetAtomicNum() == 8)
    candidates = {}
    matches = mol.GetSubstructMatches(query.GetMol(), useChirality=True,
                                     uniquify=False, maxMatches=10000)
    if len(matches) >= 10000:
        raise ValueError('Mapped linker hint matching reached the search limit; provide a more specific hint.')
    for match in matches:
        core = {match[i] for i in core_ids}
        if core & backbone:
            continue
        owners = {}
        for position, residue in enumerate(residues, 1):
            ca = residue['CA'];queue = [ca];seen = {ca}
            while queue:
                current = queue.pop()
                for atom in mol.GetAtomWithIdx(current).GetNeighbors():
                    idx = atom.GetIdx()
                    if idx in seen or idx in core or idx in backbone:
                        continue
                    seen.add(idx);owners.setdefault(idx, set()).add(position);queue.append(idx)
        attachments = []
        for dummy in dummies:
            target = match[dummy.GetIdx()]
            label = dummy.GetAtomMapNum()
            if target in backbone or target in core or owners.get(target) != {label}:
                break
            attachments.append((label, match[dummy.GetNeighbors()[0].GetIdx()], target))
        else:
            actual = {(idx, a.GetIdx()) for idx in core for a in mol.GetAtomWithIdx(idx).GetNeighbors() if a.GetIdx() not in core}
            requested = {(inside, outside) for _, inside, outside in attachments}
            if actual != requested:
                continue
            key = (frozenset(core), tuple(sorted(attachments)))
            candidates[key] = (core, attachments)
    if not candidates:
        raise ValueError('Mapped linker hint does not match the input graph at the specified residue positions and complete attachment boundaries.')
    if len(candidates) > 1:
        import warnings
        message = f'Ambiguous mapped linker hint: {len(candidates)} distinct attachments satisfy the supplied maps; provide a more specific linker.'
        warnings.warn(message, AttachmentSelectionWarning, stacklevel=2)
        raise ValueError(message)
    core, attachments = next(iter(candidates.values()))
    # Export the actual target fragment, restoring terminal mapped dummies.
    boundary_labels = {(inside, outside): position for position, inside, outside in attachments}
    bond_ids = [mol.GetBondBetweenAtoms(inside, outside).GetIdx()
                for _, inside, outside in attachments]
    # FragmentOnBonds preserves bond/atom stereo while replacing the real
    # attachment bonds; adding extra bonds to the intact core would overfill it.
    fragment = Chem.FragmentOnBonds(mol, bond_ids, addDummies=True)
    new_dummies = []
    for atom in fragment.GetAtoms():
        if atom.GetIdx() < mol.GetNumAtoms() or atom.GetAtomicNum() != 0:
            continue
        inside = atom.GetNeighbors()[0].GetIdx()
        if inside not in core:
            continue
        outside = atom.GetIsotope()
        atom.SetAtomMapNum(boundary_labels[(inside, outside)])
        atom.SetIsotope(0)
        new_dummies.append(atom.GetIdx())
    mapped = Chem.MolFragmentToSmiles(fragment, atomsToUse=sorted(core | set(new_dummies)), isomericSmiles=True)
    details = dict(selection_rule='exact_mapped_linker_boundary',
                   attachment_sites=[dict(residue_position=p, linker_atom_index=i,
                                          residue_atom_index=o,
                                          residue_atom_element=mol.GetAtomWithIdx(o).GetSymbol())
                                     for p, i, o in sorted(attachments)])
    return (core, sorted(labels), mapped), details
