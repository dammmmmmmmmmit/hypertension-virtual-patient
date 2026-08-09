"""REFERENCE ONLY - reconstructed. Match against your real drug_registry.py."""

HYPERTENSION_DRUGS = [
    {"name": "lisinopril", "drug_class": "ace_inhibitor"},
    {"name": "enalapril", "drug_class": "ace_inhibitor"},
    {"name": "ramipril", "drug_class": "ace_inhibitor"},
    {"name": "losartan", "drug_class": "arb"},
    {"name": "valsartan", "drug_class": "arb"},
    {"name": "irbesartan", "drug_class": "arb"},
    {"name": "metoprolol", "drug_class": "beta_blocker"},
    {"name": "atenolol", "drug_class": "beta_blocker"},
    {"name": "bisoprolol", "drug_class": "beta_blocker"},
    {"name": "amlodipine", "drug_class": "calcium_channel_blocker"},
    {"name": "nifedipine", "drug_class": "calcium_channel_blocker"},
    {"name": "diltiazem", "drug_class": "calcium_channel_blocker"},
    {"name": "hydrochlorothiazide", "drug_class": "thiazide_diuretic"},
    {"name": "chlorthalidone", "drug_class": "thiazide_diuretic"},
    {"name": "indapamide", "drug_class": "thiazide_diuretic"},
]

STANDARD_COMBINATIONS = [
    ("ace_inhibitor", "thiazide_diuretic"),
    ("arb", "thiazide_diuretic"),
    ("arb", "calcium_channel_blocker"),
    ("ace_inhibitor", "calcium_channel_blocker"),
]

DISCOURAGED_COMBINATIONS = [
    ("ace_inhibitor", "arb"),
]
