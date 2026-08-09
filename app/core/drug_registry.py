"""
Seed registry of hypertension drugs, organized by class.

IMPORTANT: We deliberately do NOT hardcode ChEMBL target IDs or compound
IDs here from memory — those need to be resolved live against the ChEMBL
API (see app/ingestion/chembl_client.py) and cached in Postgres, because
memorized IDs are a common source of silent errors. This file only
encodes things that are stable, well-known clinical facts: drug names,
their class, and the gene symbol of their primary target (which IS safe
to hardcode since it's basic pharmacology, not a database ID).

This is your Day 1-2 seed list. Expand it once ingestion is working if
you want broader coverage, but this set (5 classes x ~3 drugs) is enough
for a real combination-therapy demo.
"""

from app.schemas.compound import DrugClass

# gene_symbol is used to query ChEMBL's target search API
# (target_components__accession or target search by gene name) rather
# than trusting a hardcoded CHEMBL target ID.
HYPERTENSION_DRUGS = [
    # ACE inhibitors -> target: ACE (angiotensin converting enzyme)
    {"name": "lisinopril", "drug_class": DrugClass.ACE_INHIBITOR, "gene_symbol": "ACE"},
    {"name": "enalapril", "drug_class": DrugClass.ACE_INHIBITOR, "gene_symbol": "ACE"},
    {"name": "ramipril", "drug_class": DrugClass.ACE_INHIBITOR, "gene_symbol": "ACE"},
    # ARBs -> target: AGTR1 (angiotensin II receptor type 1)
    {"name": "losartan", "drug_class": DrugClass.ARB, "gene_symbol": "AGTR1"},
    {"name": "valsartan", "drug_class": DrugClass.ARB, "gene_symbol": "AGTR1"},
    {"name": "irbesartan", "drug_class": DrugClass.ARB, "gene_symbol": "AGTR1"},
    # Beta-blockers -> target: ADRB1 (beta-1 adrenergic receptor)
    {"name": "metoprolol", "drug_class": DrugClass.BETA_BLOCKER, "gene_symbol": "ADRB1"},
    {"name": "atenolol", "drug_class": DrugClass.BETA_BLOCKER, "gene_symbol": "ADRB1"},
    {"name": "bisoprolol", "drug_class": DrugClass.BETA_BLOCKER, "gene_symbol": "ADRB1"},
    # Calcium channel blockers -> target: CACNA1C (L-type calcium channel)
    {"name": "amlodipine", "drug_class": DrugClass.CALCIUM_CHANNEL_BLOCKER, "gene_symbol": "CACNA1C"},
    {"name": "nifedipine", "drug_class": DrugClass.CALCIUM_CHANNEL_BLOCKER, "gene_symbol": "CACNA1C"},
    {"name": "diltiazem", "drug_class": DrugClass.CALCIUM_CHANNEL_BLOCKER, "gene_symbol": "CACNA1C"},
    # Thiazide diuretics -> target: SLC12A3 (Na-Cl cotransporter, NCC)
    {"name": "hydrochlorothiazide", "drug_class": DrugClass.THIAZIDE_DIURETIC, "gene_symbol": "SLC12A3"},
    {"name": "chlorthalidone", "drug_class": DrugClass.THIAZIDE_DIURETIC, "gene_symbol": "SLC12A3"},
    {"name": "indapamide", "drug_class": DrugClass.THIAZIDE_DIURETIC, "gene_symbol": "SLC12A3"},
]

# Known clinically-standard combination pairings — used both to generate
# sensible training examples for the "combination" case and as the
# comparator set for the report's "vs existing drugs/combos" section.
STANDARD_COMBINATIONS = [
    (DrugClass.ACE_INHIBITOR, DrugClass.THIAZIDE_DIURETIC),
    (DrugClass.ARB, DrugClass.THIAZIDE_DIURETIC),
    (DrugClass.ARB, DrugClass.CALCIUM_CHANNEL_BLOCKER),
    (DrugClass.ACE_INHIBITOR, DrugClass.CALCIUM_CHANNEL_BLOCKER),
]

# Class combinations to actively flag as clinically discouraged /
# dangerous in the report generator (e.g. dual RAAS blockade risk of
# hyperkalemia). Not exhaustive — expand if you have time.
DISCOURAGED_COMBINATIONS = [
    (DrugClass.ACE_INHIBITOR, DrugClass.ARB),  # dual RAAS blockade - hyperkalemia/renal risk
]
