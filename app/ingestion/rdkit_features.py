"""
RDKit structural descriptors, computed directly from the canonical SMILES
ChEMBL already returns on `molecule/{id}.json` (`molecule_structures.canonical_smiles`).

Per the continuation brief: no separate PubChem round-trip is needed just
for descriptors — RDKit computes them locally from SMILES. PubChem is
still used elsewhere (pubchem_client.py) but only for CID resolution, which
SIDER linkage needs and RDKit can't provide.

These are model FEATURES, not labels — they describe the compound's shape/
polarity/size, which are plausible (if indirect) drivers of bioavailability
and off-target binding, not a direct efficacy signal on their own.
"""

from typing import Optional

from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


class MolecularDescriptors(BaseModel):
    molecular_weight: float
    logp: float
    tpsa: float
    h_bond_donors: int
    h_bond_acceptors: int
    rotatable_bonds: int
    aromatic_rings: int
    num_ro5_violations: int


def compute_descriptors(canonical_smiles: str) -> Optional[MolecularDescriptors]:
    """Parse a SMILES string and compute the descriptor set. Returns None
    if RDKit can't parse the SMILES (should be rare for ChEMBL-sourced
    strings, but don't assume — malformed/edge-case structures happen)."""
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    # Lipinski's rule-of-five violation count — cheap, standard "druglikeness"
    # summary feature; a rough sanity signal, not used as a hard filter.
    ro5_violations = sum(
        [
            mw > 500,
            logp > 5,
            hbd > 5,
            hba > 10,
        ]
    )

    return MolecularDescriptors(
        molecular_weight=round(mw, 2),
        logp=round(logp, 2),
        tpsa=round(rdMolDescriptors.CalcTPSA(mol), 2),
        h_bond_donors=hbd,
        h_bond_acceptors=hba,
        rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol),
        aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
        num_ro5_violations=ro5_violations,
    )


if __name__ == "__main__":
    # Lisinopril, verified canonical_smiles pulled live from ChEMBL
    # (CHEMBL1237) during development of this module.
    smiles = "NCCCC[C@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O"
    print(compute_descriptors(smiles))
