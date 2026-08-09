"""
Thin async client over the ChEMBL REST API.

NOTE: this sandbox's network egress allowlist doesn't include
www.ebi.ac.uk, so these calls are UNTESTED here — you'll need to run and
debug this on your own machine (which should have normal internet
access). Treat this as a solid first draft, not verified-working code.
Print/inspect raw responses the first time you run each function before
trusting the parsing.

Docs: https://www.ebi.ac.uk/chembl/api/data/docs
"""

import asyncio
from typing import Optional

import httpx

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"


class ChEMBLClient:
    def __init__(self, timeout: float = 30.0):
        self._client = httpx.AsyncClient(base_url=BASE_URL, timeout=timeout)

    async def close(self):
        await self._client.aclose()

    async def search_target_by_gene(self, gene_symbol: str) -> Optional[dict]:
        """Resolve a gene symbol (e.g. 'ACE', 'AGTR1') to a ChEMBL target
        record. Uses the ChEMBL full-text search endpoint rather than a
        hardcoded UniProt accession, so it should be robust to slightly
        fuzzy/partial matches. Returns the top-ranked human single-protein
        target if found.
        """
        resp = await self._client.get(
            "/target/search.json",
            params={"q": gene_symbol, "format": "json"},
        )
        resp.raise_for_status()
        data = resp.json()
        targets = data.get("targets", [])
        # Prefer exact single-protein human targets
        for t in targets:
            if (
                t.get("target_type") == "SINGLE PROTEIN"
                and t.get("organism") == "Homo sapiens"
            ):
                return t
        return targets[0] if targets else None

    async def search_molecule_by_name(self, name: str) -> Optional[dict]:
        """Resolve a drug/compound name to a ChEMBL molecule record.

        Tries an EXACT pref_name match first, falling back to the fuzzy
        full-text search only if no exact match exists. This matters more
        than it looks: ChEMBL's full-text search ranking is NOT reliably
        "exact name first" — verified live (2026-07-28) that `q=losartan`
        returns CHEMBL382821 ("LOSARTAN NITROOXY ESTER", a distinct
        NO-donating derivative compound with its own SMILES and its own
        molecule_hierarchy, i.e. NOT a salt/hydrate of losartan) ranked
        ABOVE CHEMBL191 (the actual "LOSARTAN" entry). The usual
        salt-form pattern (search hit -> parent_chembl_id recovers the
        base compound) does NOT rescue this case, because the top hit
        isn't a salt of the target drug at all — it's a different
        molecule. Blindly taking `molecules[0]` silently resolved
        "losartan" to the wrong compound for potency, SMILES, and every
        downstream feature. See DECISIONS.md #5.
        """
        exact_resp = await self._client.get(
            "/molecule.json",
            params={"pref_name__iexact": name, "format": "json"},
        )
        exact_resp.raise_for_status()
        exact_molecules = exact_resp.json().get("molecules", [])
        if exact_molecules:
            return exact_molecules[0]

        resp = await self._client.get(
            "/molecule/search.json",
            params={"q": name, "format": "json"},
        )
        resp.raise_for_status()
        data = resp.json()
        molecules = data.get("molecules", [])
        return molecules[0] if molecules else None

    async def get_bioactivities(
        self,
        molecule_chembl_id: str,
        target_chembl_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Fetch bioactivity records (IC50/Ki/EC50 etc.) for a molecule,
        optionally filtered to a specific target."""
        params = {
            "molecule_chembl_id": molecule_chembl_id,
            "format": "json",
            "limit": limit,
        }
        if target_chembl_id:
            params["target_chembl_id"] = target_chembl_id

        resp = await self._client.get("/activity.json", params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("activities", [])

    async def get_molecule_properties(self, molecule_chembl_id: str) -> Optional[dict]:
        """Fetch molecule properties (MW, LogP/ALogP, PSA, etc.) —
        used as ML features alongside/instead of RDKit-computed ones."""
        resp = await self._client.get(f"/molecule/{molecule_chembl_id}.json")
        resp.raise_for_status()
        return resp.json()

    async def get_mechanisms(self, molecule_chembl_id: str) -> list[dict]:
        """Fetch mechanism-of-action records (mechanism_of_action text,
        action_type, target_chembl_id) for the RAG index. NOTE: verified
        live that this is attached to the specific molecule_chembl_id
        `search_molecule_by_name()` returns, NOT reliably to its
        `parent_chembl_id` — opposite of the bioactivity lesson. E.g.
        lisinopril: zero mechanism records under the parent CHEMBL1237
        ("LISINOPRIL ANHYDROUS"), one record under CHEMBL419213
        ("LISINOPRIL", the exact-name search hit). Callers should try the
        search-hit ID first and fall back to parent_chembl_id, not the
        other way around."""
        resp = await self._client.get(
            "/mechanism.json",
            params={"molecule_chembl_id": molecule_chembl_id, "format": "json"},
        )
        resp.raise_for_status()
        return resp.json().get("mechanisms", [])


async def _smoke_test():
    """Run this manually on your machine (not in this sandbox) to sanity
    check the client: `python -m app.ingestion.chembl_client`"""
    client = ChEMBLClient()
    try:
        target = await client.search_target_by_gene("ACE")
        print("Target search (ACE):", target["target_chembl_id"], target["pref_name"])

        molecule = await client.search_molecule_by_name("lisinopril")
        print("Molecule search (lisinopril):", molecule["molecule_chembl_id"])
    
        if molecule:
            activities = await client.get_bioactivities(
                molecule["molecule_chembl_id"],
                target_chembl_id=target["target_chembl_id"],
            )
            print(f"Found {len(activities)} ACE-specific bioactivity records")
            for a in activities[:5]:
                print(a["standard_type"], a["standard_value"], a["standard_units"], "| pchembl:", a["pchembl_value"])
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(_smoke_test())
