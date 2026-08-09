"""
SIDER side-effect ingestion. Not a live API — SIDER ships as flat TSV dumps
(http://sideeffects.embl.de/download/), so this module downloads them once
into data/raw/sider/ and parses locally rather than re-fetching per query.

ID mapping — verified, then REVISED after a real discrepancy found during
development (2026-07-28), see DECISIONS.md #2 for the full writeup:

Original plan was to resolve one PubChem CID per drug ourselves
(pubchem_client.py) and join against meddra_all_se.tsv column 1 (flat
STITCH ID = "CID1" + PubChem_CID zero-padded). This broke on lisinopril:
PubChem's name search returns CID 5362119 (the specific stereoisomer,
titled "Lisinopril"), but SIDER's drug_names.tsv uses CID 3937 (the
flat/achiral parent structure, same molecular formula, no stereo
descriptors) for the same drug. PubChem assigns many CIDs to what's
clinically "one drug" (salts, stereoisomers, protonation states) and
there's no single reliable rule to pick the one an external database
picked years ago — `cids_type=same_connectivity` returns 20+ candidate
CIDs for lisinopril alone, unranked.

**Fix: join on drug name against SIDER's own drug_names.tsv (case-
insensitive exact match), not on an independently-resolved PubChem CID.**
This sidesteps the CID-ambiguity problem entirely — SIDER's own file tells
us exactly which STITCH ID it used for a given generic name. We still
resolve a PubChem CID separately (pubchem_client.py) for other purposes,
but it is NOT used as the SIDER join key.

Confirmed gap during development: 14 of the 15 registry drugs match by
name in drug_names.tsv; **enalapril has no entry at all** (checked by name
and, prior to this fix, was also absent by its resolved PubChem CID — a
genuine SIDER coverage gap, not a mapping bug). Treat this the same way as
the thiazide potency gap: leave enalapril's side-effect labels
empty/missing, disclose it, don't substitute another drug's data or a
class average silently.

Side effects are recorded per LLT (lowest-level term) AND its parent PT
(preferred term); per the SIDER README, LLTs are often near-duplicates of
their PT (e.g. "Creatinine increased" / "Blood creatinine increased" both
roll up to PT "Blood creatinine increased"). We use PT-level rows only —
using LLT would inflate/fragment the same clinical concept into several
label columns.
"""

import gzip
import shutil
from pathlib import Path
from typing import Optional

import httpx

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "sider"
SE_FILE = DATA_DIR / "meddra_all_se.tsv.gz"
SE_URL = "http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz"
NAMES_FILE = DATA_DIR / "drug_names.tsv"
NAMES_URL = "http://sideeffects.embl.de/media/download/drug_names.tsv"

FLAT_STITCH_OFFSET = 100_000_000

_name_to_flat_id_cache: Optional[dict[str, str]] = None


def download_sider_files(force: bool = False) -> None:
    """Fetch the SIDER side-effect + drug-name dumps into data/raw/sider/
    if not already present. Not called automatically on import — ingestion
    scripts call this explicitly so it's obvious when a network fetch
    happens."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60.0) as client:
        if force or not SE_FILE.exists():
            resp = client.get(SE_URL)
            resp.raise_for_status()
            SE_FILE.write_bytes(resp.content)
        if force or not NAMES_FILE.exists():
            resp = client.get(NAMES_URL)
            resp.raise_for_status()
            NAMES_FILE.write_bytes(resp.content)


def _load_name_to_flat_id() -> dict[str, str]:
    """Lowercased drug name -> flat STITCH ID, from drug_names.tsv. This is
    the actual join key SIDER used when it was built — see module
    docstring for why we match on this instead of an independently
    re-resolved PubChem CID."""
    global _name_to_flat_id_cache
    if _name_to_flat_id_cache is not None:
        return _name_to_flat_id_cache
    if not NAMES_FILE.exists():
        raise FileNotFoundError(f"{NAMES_FILE} not found — call download_sider_files() first")

    mapping: dict[str, str] = {}
    with open(NAMES_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            flat_id, name = parts
            mapping[name.lower()] = flat_id
    _name_to_flat_id_cache = mapping
    return mapping


def get_flat_stitch_id_for_name(drug_name: str) -> Optional[str]:
    """Case-insensitive exact match against SIDER's own drug_names.tsv.
    Returns None if the drug isn't in SIDER at all (a real, disclosed gap
    — e.g. enalapril — not a lookup bug)."""
    return _load_name_to_flat_id().get(drug_name.lower())


def pubchem_cid_to_flat_stitch_id(pubchem_cid: int) -> str:
    """CID1 + PubChem CID zero-padded to 8 digits, matching column 1 of
    meddra_all_se.tsv exactly (string form, since that's what's in the file)."""
    return f"CID1{pubchem_cid:08d}"


def _iter_se_rows():
    """Yield (flat_stitch_id, concept_type, side_effect_name) tuples for
    every row in the local meddra_all_se.tsv.gz. Raises FileNotFoundError
    if download_sider_files() hasn't been run yet."""
    if not SE_FILE.exists():
        raise FileNotFoundError(f"{SE_FILE} not found — call download_sider_files() first")
    with gzip.open(SE_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 6:
                continue  # malformed line — skip rather than crash the whole parse
            flat_id, _stereo_id, _umls_label, concept_type, _umls_meddra, se_name = parts
            yield flat_id, concept_type, se_name


def get_side_effects_by_flat_id(flat_stitch_id: str) -> list[str]:
    """Distinct PT-level side-effect names for a given flat STITCH ID."""
    side_effects = set()
    for flat_id, concept_type, se_name in _iter_se_rows():
        if flat_id == flat_stitch_id and concept_type == "PT":
            side_effects.add(se_name)
    return sorted(side_effects)


def get_side_effects_for_drug(drug_name: str) -> list[str]:
    """Distinct PT-level side-effect names reported for a drug, matched by
    name against SIDER's own drug_names.tsv. Returns [] if the drug has no
    SIDER coverage (a real, disclosed gap — e.g. enalapril, see module
    docstring) rather than raising, since "no data" is an expected,
    meaningful outcome here."""
    flat_id = get_flat_stitch_id_for_name(drug_name)
    if flat_id is None:
        return []
    return get_side_effects_by_flat_id(flat_id)


def get_side_effects_for_cid(pubchem_cid: int) -> list[str]:
    """Fallback path: look up by an independently-resolved PubChem CID
    directly. NOT the primary join — see module docstring for why this can
    silently miss (PubChem assigns multiple CIDs per drug; SIDER may have
    used a different one than whatever we resolved). Prefer
    get_side_effects_for_drug() wherever the drug name is known."""
    target_flat_id = pubchem_cid_to_flat_stitch_id(pubchem_cid)
    return get_side_effects_by_flat_id(target_flat_id)


def build_side_effect_vocabulary(drug_names: list[str], top_n: int = 30) -> list[str]:
    """Given the registry's drug names, return the top_n most frequently
    reported PT-level side effects across all of them — this fixed-width
    list becomes the multi-label target vector for the side-effect
    prediction task."""
    from collections import Counter

    counts: Counter = Counter()
    for name in drug_names:
        for se in get_side_effects_for_drug(name):
            counts[se] += 1
    return [name for name, _ in counts.most_common(top_n)]


def cleanup_source_gz() -> None:
    """SIDER's dump is CC-BY-SA 4.0 (attribution required, see
    data/raw/sider — keep the file, don't delete it silently); this helper
    exists only to decompress a persistent .tsv copy if a downstream tool
    needs uncompressed input. Not called by default."""
    if not SE_FILE.exists():
        return
    out_path = SE_FILE.with_suffix("")  # drops .gz
    with gzip.open(SE_FILE, "rb") as src, open(out_path, "wb") as dst:
        shutil.copyfileobj(src, dst)


if __name__ == "__main__":
    """Run: uv run python -m app.ingestion.sider_client"""
    download_sider_files()

    print("lisinopril side effects (PT):", get_side_effects_for_drug("lisinopril")[:10])
    print("losartan side effects (PT):", get_side_effects_for_drug("losartan")[:10])
    print("enalapril side effects (PT):", get_side_effects_for_drug("enalapril"), "<- confirmed gap")
