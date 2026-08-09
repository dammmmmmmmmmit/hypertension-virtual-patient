"""
Loop over HYPERTENSION_DRUGS, resolve each drug's parent ChEMBL ID and
its class's target ChEMBL ID, fetch target-specific bioactivity, and
print a summary table of valid pchembl_value coverage per drug.

Key lesson baked in here (learned via debug_scratch.py): always fetch
bioactivity using the PARENT chembl_id, not the raw molecule_chembl_id
returned by molecule search -- salt/hydrate forms often have no directly
attached bioactivity data, it's attached to the parent compound.

Run: uv run python -m app.ingestion.build_registry
"""

import asyncio
import statistics

from app.core.drug_registry import HYPERTENSION_DRUGS
from app.ingestion.chembl_client import ChEMBLClient
from app.ingestion.potency_utils import extract_best_potency


async def build_gene_target_cache(client: ChEMBLClient) -> dict[str, str | None]:
    """Resolve each unique gene_symbol to a target_chembl_id once,
    rather than once per drug (5 unique genes vs 15 drugs)."""
    unique_genes = sorted({d["gene_symbol"] for d in HYPERTENSION_DRUGS})
    cache: dict[str, str | None] = {}

    for gene in unique_genes:
        target = await client.search_target_by_gene(gene)
        if target:
            cache[gene] = target["target_chembl_id"]
            print(f"Resolved target {gene} -> {target['target_chembl_id']} ({target['pref_name']})")
        else:
            cache[gene] = None
            print(f"WARNING: could not resolve target for gene {gene}")
        await asyncio.sleep(0.3)

    return cache


async def resolve_drug(client: ChEMBLClient, drug: dict, target_id) -> dict:
    """Resolve one drug's parent ChEMBL ID and fetch target-specific
    bioactivity, returning a summary row."""
    row = {
        "name": drug["name"],
        "drug_class": drug["drug_class"].value,
        "gene_symbol": drug["gene_symbol"],
        "target_chembl_id": target_id,
        "parent_chembl_id": None,
        "n_valid_records": 0,
        "mean_pchembl": None,
        "note": "",
    }

    molecule = await client.search_molecule_by_name(drug["name"])
    if not molecule:
        row["note"] = "molecule not found"
        return row

    hierarchy = molecule.get("molecule_hierarchy") or {}
    parent_id = hierarchy.get("parent_chembl_id") or molecule.get("molecule_chembl_id")
    row["parent_chembl_id"] = parent_id

    if not target_id:
        row["note"] = "no target resolved for this gene"
        return row

    activities = await client.get_bioactivities(parent_id, target_chembl_id=target_id)

    # Prefer ChEMBL's own pchembl_value; fall back to manual pX computation
    # for records with a usable standard_value/units/relation but a
    # standard_type ChEMBL doesn't auto-transform (e.g. 'Kb' - see losartan).
    potency_values = [extract_best_potency(a) for a in activities]
    valid_values = [v for v in potency_values if v is not None]
    row["n_valid_records"] = len(valid_values)

    if valid_values:
        row["mean_pchembl"] = round(statistics.mean(valid_values), 2)
    else:
        row["note"] = "0 valid records (no pchembl_value AND no usable exact-relation fallback)"

    return row


async def main():
    client = ChEMBLClient()
    try:
        print("=== Resolving gene -> target cache ===")
        gene_target_cache = await build_gene_target_cache(client)

        print("\n=== Resolving drugs + fetching bioactivity ===")
        results = []
        for drug in HYPERTENSION_DRUGS:
            target_id = gene_target_cache.get(drug["gene_symbol"])
            row = await resolve_drug(client, drug, target_id)
            results.append(row)
            flag = "OK" if row["n_valid_records"] > 0 else "WARN"
            print(
                f"[{flag}] {row['name']:20s} class={row['drug_class']:28s} "
                f"parent={str(row['parent_chembl_id']):15s} target={str(row['target_chembl_id']):15s} "
                f"n={row['n_valid_records']:3d} mean_pchembl={row['mean_pchembl']}"
                + (f"  <- {row['note']}" if row["note"] else "")
            )
            await asyncio.sleep(0.3)

        print("\n=== Summary ===")
        zero_coverage = [r for r in results if r["n_valid_records"] == 0]
        print(f"Total drugs: {len(results)}")
        print(f"Drugs with zero valid records: {len(zero_coverage)}")
        if zero_coverage:
            print("Names:", [r["name"] for r in zero_coverage])

        return results
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())