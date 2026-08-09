"""
Thin async client over PubChem's PUG-REST API. Used for exactly one thing:
resolving a generic drug name to a PubChem CID, which is the join key SIDER
needs (see DECISIONS.md #2 for the STITCH-flat-ID <-> PubChem-CID mapping).

Structural descriptors are NOT fetched from here — RDKit computes those
locally from the canonical_smiles ChEMBL already returns (rdkit_features.py).
This client's only job is the CID lookup.

Verified reachable and returns sane results during development:
`GET /rest/pug/compound/name/lisinopril/cids/JSON` -> CID 5362119 (matches
the well-known PubChem CID for lisinopril).
"""

import asyncio
from typing import Optional

import httpx

BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


class PubChemClient:
    def __init__(self, timeout: float = 30.0):
        self._client = httpx.AsyncClient(base_url=BASE_URL, timeout=timeout)

    async def close(self):
        await self._client.aclose()

    async def get_cid_by_name(self, name: str) -> Optional[int]:
        """Resolve a drug/compound name to a single PubChem CID. Name search
        against PubChem typically returns the canonical parent compound as
        the first (often only) hit for a generic drug name — unlike ChEMBL's
        molecule search, which frequently returns a specific salt form.
        """
        resp = await self._client.get(f"/compound/name/{name}/cids/JSON")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        cids = data.get("IdentifierList", {}).get("CID", [])
        return cids[0] if cids else None

    async def get_cid_by_inchikey(self, inchikey: str) -> Optional[int]:
        """Fallback path if name search fails (e.g. brand vs. INN name
        mismatch) — exact structure match via InChIKey, using the
        standard_inchi_key ChEMBL already provides. Returns the first CID;
        InChIKey lookups can return multiple related entries (parent +
        specific salts), so this is a fallback, not the primary path."""
        resp = await self._client.get(f"/compound/inchikey/{inchikey}/cids/JSON")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        cids = data.get("IdentifierList", {}).get("CID", [])
        return cids[0] if cids else None


async def _smoke_test():
    """Run: uv run python -m app.ingestion.pubchem_client"""
    client = PubChemClient()
    try:
        cid = await client.get_cid_by_name("lisinopril")
        print("lisinopril CID (by name):", cid)

        cid2 = await client.get_cid_by_inchikey("RLAWWYSOJDYHDC-BZSNNMDCSA-N")
        print("lisinopril CID (by InChIKey, first hit):", cid2)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(_smoke_test())
