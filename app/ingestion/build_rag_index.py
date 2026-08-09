"""
Builds the Qdrant RAG index over mechanism-of-action text for the 15
registry drugs, used by the report generator (Week 2) to ground its
mechanism summaries in retrieved text rather than the LLM's own
(unverifiable, possibly hallucinated) recall.

Source: ChEMBL's `/mechanism.json?molecule_chembl_id=X` — per the
continuation brief's own suggestion, this removes the DrugBank
academic-license dependency entirely. Verified live (2026-07-28): direct
coverage for 9/15 drugs. NOTE the mechanism data is keyed to the specific
molecule_chembl_id `search_molecule_by_name()` returns, NOT reliably to
`parent_chembl_id` (opposite of the bioactivity lesson) — see
`ChEMBLClient.get_mechanisms()` docstring.

The other 6 (enalapril, losartan, metoprolol, bisoprolol, amlodipine,
diltiazem) have no ChEMBL mechanism record at all. Per this project's
consistent policy (Critical Engineering Decision #4 and DECISIONS.md #1)
we do NOT fabricate a compound-specific mechanism sentence for these —
each RAG document is tagged with `source`: "chembl_mechanism" (real,
compound-specific ChEMBL text) or "class_level_fallback" (a short,
templated description built only from the class/gene facts already
hardcoded in drug_registry.py, which are safe, stable pharmacology, not
invented). `generate_report` (Week 2) must disclose which kind of source
backed each mechanism sentence it used — never present a fallback
description as if it were a ChEMBL-sourced record.

Embeddings: FastEmbed (local, ONNX, no external API key needed) via
qdrant-client's built-in `.add()`/`.query()` convenience methods, default
model BAAI/bge-small-en. Chosen because it's a natural, already-declared-
dependency-adjacent complement to qdrant-client (see pyproject.toml) and
needs no additional API key — consistent with keeping the RAG layer
runnable without extra account setup.

Run: uv run python -m app.ingestion.build_rag_index
"""

import asyncio

from qdrant_client import QdrantClient

from app.core.drug_registry import HYPERTENSION_DRUGS
from app.core.settings import settings
from app.ingestion.chembl_client import ChEMBLClient

COLLECTION_NAME = "drug_mechanisms"

CLASS_LEVEL_DESCRIPTIONS = {
    "ace_inhibitor": "inhibits angiotensin-converting enzyme (ACE), reducing conversion of "
                      "angiotensin I to angiotensin II and lowering vasoconstriction/aldosterone release",
    "arb": "blocks the type-1 angiotensin II receptor (AGTR1), preventing angiotensin II from "
           "causing vasoconstriction and aldosterone release",
    "beta_blocker": "antagonizes beta-1 adrenergic receptors, reducing heart rate, cardiac "
                     "contractility, and renin release",
    "calcium_channel_blocker": "blocks voltage-gated L-type calcium channels in vascular smooth "
                                "muscle, causing vasodilation",
    "thiazide_diuretic": "inhibits the sodium-chloride cotransporter (NCC) in the distal "
                          "convoluted tubule, increasing sodium/water excretion",
}


async def resolve_mechanism_text(client: ChEMBLClient, drug: dict) -> tuple[str, str]:
    """Returns (mechanism_text, source) where source is
    'chembl_mechanism' or 'class_level_fallback'."""
    name = drug["name"]
    mol = await client.search_molecule_by_name(name)
    if mol:
        mechs = await client.get_mechanisms(mol["molecule_chembl_id"])
        if mechs and mechs[0].get("mechanism_of_action"):
            action_type = mechs[0].get("action_type", "")
            text = mechs[0]["mechanism_of_action"]
            if action_type:
                text = f"{text} (action type: {action_type})"
            return text, "chembl_mechanism"

    fallback = CLASS_LEVEL_DESCRIPTIONS.get(drug["drug_class"].value, "mechanism not characterized in this registry")
    return f"{name.title()} is a {drug['drug_class'].value.replace('_', ' ')}: it {fallback}.", "class_level_fallback"


async def main():
    client = ChEMBLClient()
    qdrant = QdrantClient(url=settings.qdrant_url)

    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)

    documents = []
    metadata = []
    ids = []

    try:
        for i, drug in enumerate(HYPERTENSION_DRUGS):
            name = drug["name"]
            text, source = await resolve_mechanism_text(client, drug)
            doc = f"{name.title()} ({drug['drug_class'].value.replace('_', ' ')}): {text}"
            documents.append(doc)
            metadata.append({
                "drug_name": name,
                "drug_class": drug["drug_class"].value,
                "gene_symbol": drug["gene_symbol"],
                "mechanism_text": text,
                "source": source,
            })
            ids.append(i)
            print(f"[{source:20s}] {name:20s} {text[:80]}")
            await asyncio.sleep(0.25)
    finally:
        await client.close()

    qdrant.add(collection_name=COLLECTION_NAME, documents=documents, metadata=metadata, ids=ids)
    print(f"\nIndexed {len(documents)} mechanism documents into Qdrant collection '{COLLECTION_NAME}'")

    n_chembl = sum(1 for m in metadata if m["source"] == "chembl_mechanism")
    print(f"  {n_chembl}/{len(metadata)} from real ChEMBL mechanism records, "
          f"{len(metadata) - n_chembl} from class-level fallback (disclosed, not fabricated)")


if __name__ == "__main__":
    asyncio.run(main())
