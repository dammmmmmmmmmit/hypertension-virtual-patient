"""
Full ingestion pass: for each HYPERTENSION_DRUGS entry, resolve ChEMBL
(reusing build_registry.py's parent-ID + potency-fallback logic — don't
duplicate that here), PubChem CID, RDKit descriptors, and SIDER side
effects, then upsert one row per drug into `resolved_compounds`.

This is the thing that makes the rest of the pipeline NOT re-hit external
APIs on every run — build_registry.py is a read-only diagnostic/coverage
report; this script is what actually populates the cache it reports on.

Run: uv run python -m app.ingestion.populate_cache
"""

import asyncio

from sqlalchemy import select

from app.core.drug_registry import HYPERTENSION_DRUGS
from app.db.models import ResolvedCompound
from app.db.session import async_session_factory
from app.ingestion.build_registry import build_gene_target_cache, resolve_drug
from app.ingestion.chembl_client import ChEMBLClient
from app.ingestion.pubchem_client import PubChemClient
from app.ingestion.rdkit_features import compute_descriptors
from app.ingestion.sider_client import download_sider_files, get_side_effects_for_drug


async def resolve_pubchem_cid(pubchem_client: PubChemClient, name: str, inchikey: str | None) -> int | None:
    cid = await pubchem_client.get_cid_by_name(name)
    if cid is None and inchikey:
        cid = await pubchem_client.get_cid_by_inchikey(inchikey)
    return cid


async def main():
    download_sider_files()  # no-op if already cached locally

    chembl = ChEMBLClient()
    pubchem = PubChemClient()

    try:
        print("=== Resolving gene -> target cache ===")
        gene_target_cache = await build_gene_target_cache(chembl)

        async with async_session_factory() as session:
            for drug in HYPERTENSION_DRUGS:
                name = drug["name"]
                target_id = gene_target_cache.get(drug["gene_symbol"])

                chembl_row = await resolve_drug(chembl, drug, target_id)
                await asyncio.sleep(0.3)

                # Fetch full molecule record for canonical_smiles + InChIKey
                # (needed for RDKit descriptors and the PubChem fallback path).
                canonical_smiles = None
                inchikey = None
                if chembl_row["parent_chembl_id"]:
                    mol = await chembl.get_molecule_properties(chembl_row["parent_chembl_id"])
                    structures = (mol or {}).get("molecule_structures") or {}
                    canonical_smiles = structures.get("canonical_smiles")
                    inchikey = structures.get("standard_inchi_key")
                await asyncio.sleep(0.3)

                descriptors = compute_descriptors(canonical_smiles) if canonical_smiles else None

                pubchem_cid = await resolve_pubchem_cid(pubchem, name, inchikey)
                await asyncio.sleep(0.3)

                side_effects = get_side_effects_for_drug(name)

                flag = "OK" if chembl_row["n_valid_records"] > 0 else "GAP(potency)"
                se_flag = f"{len(side_effects)} SE" if side_effects else "GAP(SIDER)"
                print(f"[{flag:14s}][{se_flag:10s}] {name:20s} cid={pubchem_cid} smiles={'yes' if canonical_smiles else 'no'}")

                existing = await session.scalar(select(ResolvedCompound).where(ResolvedCompound.name == name))
                row = existing or ResolvedCompound(name=name)
                row.drug_class = drug["drug_class"].value
                row.gene_symbol = drug["gene_symbol"]
                row.chembl_id = chembl_row["parent_chembl_id"]  # we standardize on parent id, see chembl_client/build_registry notes
                row.parent_chembl_id = chembl_row["parent_chembl_id"]
                row.target_chembl_id = chembl_row["target_chembl_id"]
                row.canonical_smiles = canonical_smiles
                row.mean_potency = chembl_row["mean_pchembl"]
                row.n_valid_potency_records = chembl_row["n_valid_records"]
                row.pubchem_cid = pubchem_cid
                row.side_effects = side_effects
                row.rdkit_descriptors = descriptors.model_dump() if descriptors else None

                session.add(row)

            await session.commit()

        print("\nDone — resolved_compounds table updated.")
    finally:
        await chembl.close()
        await pubchem.close()


if __name__ == "__main__":
    asyncio.run(main())
