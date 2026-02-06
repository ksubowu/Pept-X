from rdkit.Chem.rdchem import BondType
from typing import Dict, List, Optional, Sequence, Tuple
from rdkit import Chem
from rdkit.Chem import rdmolops

import os
from pathlib import Path
import rdkit
from rdkit.Chem.Draw import MolDraw2DCairo
from rdkit.Chem import AllChem
from seq2smi import MonomerLib  # pylint: disable=import-outside-toplevel
from utils import clean_smiles, get_backbone_atoms
from rdkit.Chem import rdmolfiles

# /cadd_data/cw/works/pys/molstar/build/viewer

def _find_iso_peptide(mol: Chem.Mol, backbone: Sequence[Tuple[int, int, int]]) -> Optional[Tuple[int, int]]:
    bb_atoms = {idx for triplet in backbone for idx in triplet}
    iso_peptideBond=dict()
    for tail_res_idx, triplet in enumerate(backbone):
        n_at=triplet[0]
        atom_N = mol.GetAtomWithIdx(n_at)
        for nb in atom_N.GetNeighbors():
            if nb.GetAtomicNum() == 6 and nb.GetIdx() not in bb_atoms:
                for bond in nb.GetBonds():
                    if bond.GetBondType() == BondType.DOUBLE:
                        other_atom = bond.GetOtherAtom(nb)
                        if other_atom.GetAtomicNum() == 8:  # 8 = O
                            # iso_peptideBond[tail_res_idx]=[n_at,nb.GetIdx(),other_atom.GetIdx()]#N-C(=O)
                            iso_peptideBond[tail_res_idx]=[n_at,nb.GetIdx(),other_atom.GetIdx()]#N-C(=O)
                            break   
    if len(iso_peptideBond)>0:
        print(f"iso-peptide Bond",iso_peptideBond)      
    return iso_peptideBond

# os.getcwd()
SCRIPT_DIR=Path(os.getcwd())
DEFAULT_LIB = SCRIPT_DIR / "data" / "monomersFromHELMCoreLibrary.json"
print(DEFAULT_LIB)
lib = MonomerLib(str(DEFAULT_LIB))

smi='CC[C@@H]([C@@H]1NC([C@@H]2CCCN2C([C@@H](NC([C@@H](NC([C@@H](NC([C@@H](N(C([C@@H](NC([C@@H](NC([C@@H](NC(C[C@H](NC(CNC([C@@H](NC1=O)[C@H](O)C)=O)=O)C(N3CCCCC3)=O)=O)C)=O)C)=O)CC(C)C)=O)C)C(C)C)=O)CC(C)C)=O)Cc4ccccc4)=O)Cc5ccccc5)=O)=O)C'

mol = rdkit.Chem.MolFromSmiles(smi)
backbone = get_backbone_atoms(mol)
backbone=list(backbone)

flat_set = {element for tuple_item in backbone for element in tuple_item}
result_list = list(flat_set)
# create_publication_quality_mol(mol, "publication_ready.png")



iso_peptideBond = _find_iso_peptide(mol, backbone)

# 遍历C
CcapBond=dict()
bb_atoms = {idx for triplet in backbone for idx in triplet}
for tail_res_idx, triplet in enumerate(backbone):
    c_at=triplet[-1]
    atom_C = mol.GetAtomWithIdx(c_at)
    for bond in atom_C.GetBonds():
        if bond.GetBondType() != BondType.DOUBLE: 
            other_atom = bond.GetOtherAtom(atom_C)
            if other_atom.GetIdx() not in bb_atoms:
                # Ccap_bond = mol.GetBondBetweenAtoms(c_at,other_atom.GetIdx())
                CcapBond[tail_res_idx]=[c_at,other_atom.GetIdx()]#C-N
if len(CcapBond)>0:
    print(CcapBond)

# 遍历N for Ncap if exists should exclude isoPeptideBond
NcapBond=dict()
bb_atoms = {idx for triplet in backbone for idx in triplet}
for tail_res_idx, triplet in enumerate(backbone):
    N_at=triplet[0]
    atom_ = mol.GetAtomWithIdx(N_at)
    for nb in atom_.GetNeighbors():
        if nb.GetIdx() not in bb_atoms:
            if len(iso_peptideBond)>0:
                CO_s=[v[1] for k,v in iso_peptideBond.items() ]
                if nb.GetIdx() not in CO_s:
                    NcapBond[tail_res_idx]=[N_at,nb.GetIdx()]
            else:
                NcapBond[tail_res_idx]=[N_at,nb.GetIdx()]
if len(NcapBond)>0:
    print(NcapBond)

    for bond in atom_.GetBonds():
        if bond.GetBondType() != BondType.DOUBLE: 
            other_atom = bond.GetOtherAtom(atom_)
            if other_atom.GetIdx() not in bb_atoms:
                CcapBond[tail_res_idx]=[N_at,other_atom.GetIdx()]#C-N
if len(CcapBond)>0:
    print(CcapBond)

#TODO determine order via peptide bonds (C of i to N of i+1)
prev_map: Dict[int, int] = {}
next_map: Dict[int, int] = {}

for idx, res in enumerate(backbone):
    for jdx, other in enumerate(backbone):#N-CA-C
        if idx == jdx:
            continue
        if mol.GetBondBetweenAtoms(other[0], res[-1]):
            next_map[idx] = jdx
        if mol.GetBondBetweenAtoms(other[-1], res[0]):
            prev_map[idx] = jdx

start_idx = None
for idx in range(len(backbone)):
    if idx not in prev_map:
        start_idx = idx
        break
if start_idx is None:
    # fallback: order by nitrogen index
    order = sorted(range(len(backbone)), key=lambda i: backbone[i]["N"])
else:
    order = [start_idx]
    seen = {start_idx}
    while order[-1] in next_map:
        nxt = next_map[order[-1]]
        if nxt in seen:
            break
        order.append(nxt)
        seen.add(nxt)
    if len(order) != len(backbone):
        # incomplete traversal; fallback to sorted order
        order = sorted(
            range(len(backbone)), key=lambda i: backbone[i]["N"]
        )
# reorder residues in-place for downstream processing
ordered_residues = [backbone[i] for i in order]                    
#as we know larita Ncap is the D/E's side chain, so jusst get the Ccap
# Check N-terminal nitrogen
n_cap_atoms = set()
c_cap_atoms = set()
residues = [{"N": n, "CA": ca, "C": c} for (n, ca, c) in ordered_residues]
# n_atom = mol.GetAtomWithIdx(residues[0]["N"])#not suitable for cycle or lariat, as0 may be not the first one resudes
# for nb in n_atom.GetNeighbors():
#     if (nb.GetIdx() != residues[0]["CA"] and 
#         nb.GetAtomicNum() != 1):  # Not H and not CA
#         n_cap_atoms.add(nb.GetIdx())
#         # Add connected non-backbone atoms recursively
#         stack = [nb.GetIdx()]
#         while stack:
#             idx = stack.pop()
#             atom = mol.GetAtomWithIdx(idx)
#             for next_nb in atom.GetNeighbors():
#                 next_idx = next_nb.GetIdx()
#                 if (next_idx not in n_cap_atoms and 
#                     next_idx not in [residues[0]["N"], residues[0]["CA"], residues[0]["C"]]):
#                     n_cap_atoms.add(next_idx)
#                     stack.append(next_idx)

# Check C-terminal carbonyl
c_atom = mol.GetAtomWithIdx(residues[-1]["C"])
for nb in c_atom.GetNeighbors():
    if (nb.GetIdx() != residues[-1]["CA"] and 
        not (nb.GetAtomicNum() == 8 and  # Skip C=O
                any(b.GetBondType() == Chem.BondType.DOUBLE 
                    for b in nb.GetBonds()))):
        c_cap_atoms.add(nb.GetIdx())
        # Add connected non-backbone atoms recursively
        stack = [nb.GetIdx()]
        while stack:
            idx = stack.pop()
            atom = mol.GetAtomWithIdx(idx)
            for next_nb in atom.GetNeighbors():
                next_idx = next_nb.GetIdx()
                if (next_idx not in c_cap_atoms and 
                    next_idx not in [residues[-1]["N"], residues[-1]["CA"], residues[-1]["C"]]):
                    c_cap_atoms.add(next_idx)
                    stack.append(next_idx)

cap_info={"n_cap": n_cap_atoms, "c_cap": c_cap_atoms}

bond_indices = []
for addbond in [iso_peptideBond, CcapBond]:
    for k,v in addbond.items():
        print(*v)
        a1,a2=v[0], v[1]
        b_ = mol.GetBondBetweenAtoms(a1,a2)
        if b_: bond_indices.append(b_.GetIdx()) 
#normal peptide bonds
for idx in range(len(residues) - 1):
    bond = mol.GetBondBetweenAtoms(
        residues[idx]["C"], residues[idx + 1]["N"]
    )
    if bond:
        bond_indices.append(bond.GetIdx())

# break iso_peptideBond CcapBond,NcapBond, peptideBonds| fragment to get resid unit at once
fragmol = (
            rdmolops.FragmentOnBonds(mol, bond_indices, addDummies=True)
            if bond_indices
            else Chem.Mol(mol)
        )

atom_frags = Chem.GetMolFrags(fragmol, asMols=False, sanitizeFrags=False)
# map CA atom -> fragment index
ca_to_fragment: Dict[int, int] = {}
for frag_idx, atom_ids in enumerate(atom_frags):
    for res_idx, residue in enumerate(residues):
        if residue["CA"] in atom_ids:
            ca_to_fragment[residue["CA"]] = frag_idx
if len(ca_to_fragment) != len(residues):
    raise ValueError("Failed to map fragments to residues.")

# build fragment mols (with dummies)
frag_mols = list(Chem.GetMolFrags(fragmol, asMols=True, sanitizeFrags=True))


def _canonical_mol(mol: Chem.Mol) -> Optional[str]:
    """Canonicalize a molecule in the same way for library + fragments.

    Strip atom-map numbers and explicit hydrogens so the representation
    matches how the core library stores monomers (neutral, implicit-H).
    """
    if mol is None:
        return None
    m = Chem.Mol(mol)
    for atom in m.GetAtoms():
        atom.SetAtomMapNum(0)
    m = Chem.RemoveHs(m)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)


def _canonical(smiles: str) -> Optional[str]:
    if not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    return _canonical_mol(m)


# build canonical -> code map
canon_to_code = {}
for code, entry in lib.by_code.items():
    smi = entry.get("smiles") or entry.get("smiles_L") or entry.get("smiles_D")
    if isinstance(smi, list):
        smi = next((s for s in smi if isinstance(s, str)), None)
    canon = _canonical(smi)
    if canon:
        canon_to_code[canon] = entry.get("code", code)


def cap_fragment(fragment: Chem.Mol) -> Chem.Mol:
    rw = Chem.RWMol(fragment)
    dummies = [atom.GetIdx() for atom in rw.GetAtoms() if atom.GetAtomicNum() == 0]
    for d_idx in sorted(dummies, reverse=True):
        if d_idx >= rw.GetNumAtoms():
            continue
        neighs = list(rw.GetAtomWithIdx(d_idx).GetNeighbors())
        rw.RemoveAtom(d_idx)
        for n in neighs:
            if n.GetAtomicNum() == 6:
                # ensure carbonyl has OH
                has_single_o = any(
                    nb.GetAtomicNum() == 8
                    and rw.GetBondBetweenAtoms(n.GetIdx(), nb.GetIdx()).GetBondType() == BondType.SINGLE
                    for nb in n.GetNeighbors()
                )
                if not has_single_o:
                    o_atom = Chem.Atom(8)
                    o_atom.SetNoImplicit(True)
                    o_atom.SetNumExplicitHs(1)
                    o_idx = rw.AddAtom(o_atom)
                    rw.AddBond(n.GetIdx(), o_idx, Chem.BondType.SINGLE)
            if n.GetAtomicNum() == 7:
                n.SetNoImplicit(False)
            else:
                n.SetNoImplicit(True)
            n.SetNumExplicitHs(0)
    capped = rw.GetMol()
    Chem.SanitizeMol(capped)
    return capped


def match_fragment(frag: Chem.Mol) -> Tuple[str, Optional[str]]:
    capped = cap_fragment(frag)
    canon = _canonical_mol(capped)
    code = canon_to_code.get(canon)
    return code if code else "X", canon


# order residues by peptide connectivity
order = []
if start_idx is None:
    order = sorted(range(len(residues)), key=lambda i: residues[i]["N"])
else:
    order = [start_idx]
    seen = {start_idx}
    while order[-1] in next_map:
        nxt = next_map[order[-1]]
        if nxt in seen:
            break
        order.append(nxt)
        seen.add(nxt)
    if len(order) != len(residues):
        order = sorted(range(len(residues)), key=lambda i: residues[i]["N"])

tokens = []
unmatched = []
cap_smiles = None
cap_frag_idx = None

# determine cap fragment (if any) from cap bond endpoints
if CcapBond:
    # only one C-cap expected; take the first entry
    _, (c_idx, cap_idx) = next(iter(CcapBond.items()))
    for frag_idx, atom_ids in enumerate(atom_frags):
        if cap_idx in atom_ids:
            cap_frag_idx = frag_idx
            break
    if cap_frag_idx is not None:
        cap_mol = Chem.Mol(frag_mols[cap_frag_idx])
        for atom in cap_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetIsotope(0)
        cap_smiles = _canonical_mol(cap_mol)
for idx in order:
    ca = residues[idx]["CA"]
    frag_idx = ca_to_fragment[ca]
    code, canon = match_fragment(frag_mols[frag_idx])
    tokens.append(code)
    if code == "X":
        unmatched.append({"residue_index": idx + 1, "smiles": canon})

seq_core = "-".join(tokens)
if cap_smiles:
    seq_core += f"-[C-cap:{cap_smiles}]"
sequence = seq_core + "|lariat"
print("Sequence:", sequence)
if unmatched:
    print("Unmatched fragments:")
    for entry in unmatched:
        print(entry)
