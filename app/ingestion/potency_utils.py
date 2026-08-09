"""
Fallback potency computation for bioactivity records that have a valid
numeric value/unit/relation but that ChEMBL didn't auto-compute a
pchembl_value for (e.g. standard_type='Kb', which is a legitimate
potency measure but outside the set of types ChEMBL auto-transforms).

Only use this as a FALLBACK after checking pchembl_value first -
ChEMBL's own computation is more trustworthy where it exists.
"""

import math
from typing import Optional

# Conversion factor from each unit to Molar (mol/L)
UNIT_TO_MOLAR = {
    "M": 1.0,
    "mM": 1e-3,
    "uM": 1e-6,
    "nM": 1e-9,
    "pM": 1e-12,
}


def compute_pX(value: Optional[str], units: Optional[str], relation: Optional[str]) -> Optional[float]:
    """Compute -log10(value in Molar) from a raw standard_value/units,
    mirroring what pchembl_value represents for IC50/Ki/EC50/Kd.

    Only accepts exact ('=') relations - censored values ('>', '<')
    are excluded because they don't represent a precise potency and
    would distort the training signal if treated as exact.
    """
    if value is None or units is None:
        return None
    if relation not in (None, "="):
        return None  # censored/inequality value - not usable as a point estimate
    if units not in UNIT_TO_MOLAR:
        return None

    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None

    if value_f <= 0:
        return None

    molar = value_f * UNIT_TO_MOLAR[units]
    return round(-math.log10(molar), 2)


def extract_best_potency(activity: dict) -> Optional[float]:
    """Given a single ChEMBL activity record, return the best available
    potency value: prefer ChEMBL's own pchembl_value, fall back to a
    manual pX computation from standard_value/units/relation for
    standard_types ChEMBL doesn't auto-transform (e.g. Kb)."""
    pchembl = activity.get("pchembl_value")
    if pchembl is not None:
        try:
            return float(pchembl)
        except (TypeError, ValueError):
            pass

    return compute_pX(
        activity.get("standard_value"),
        activity.get("standard_units"),
        activity.get("standard_relation"),
    )



