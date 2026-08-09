import type { DrugClass } from "./types";

/**
 * Mirrors app/core/drug_registry.py's HYPERTENSION_DRUGS exactly (15
 * drugs, 5 classes) — hardcoded here for the same reason the Python
 * backend hardcodes it: drug name/class/target-gene are stable clinical
 * facts, not database IDs that need live resolution (see that file's
 * own docstring). Read directly from the source file, not reconstructed
 * from memory.
 */
export const HYPERTENSION_DRUGS: {
  name: string;
  drugClass: DrugClass;
  geneSymbol: string;
}[] = [
  { name: "lisinopril", drugClass: "ace_inhibitor", geneSymbol: "ACE" },
  { name: "enalapril", drugClass: "ace_inhibitor", geneSymbol: "ACE" },
  { name: "ramipril", drugClass: "ace_inhibitor", geneSymbol: "ACE" },
  { name: "losartan", drugClass: "arb", geneSymbol: "AGTR1" },
  { name: "valsartan", drugClass: "arb", geneSymbol: "AGTR1" },
  { name: "irbesartan", drugClass: "arb", geneSymbol: "AGTR1" },
  { name: "metoprolol", drugClass: "beta_blocker", geneSymbol: "ADRB1" },
  { name: "atenolol", drugClass: "beta_blocker", geneSymbol: "ADRB1" },
  { name: "bisoprolol", drugClass: "beta_blocker", geneSymbol: "ADRB1" },
  { name: "amlodipine", drugClass: "calcium_channel_blocker", geneSymbol: "CACNA1C" },
  { name: "nifedipine", drugClass: "calcium_channel_blocker", geneSymbol: "CACNA1C" },
  { name: "diltiazem", drugClass: "calcium_channel_blocker", geneSymbol: "CACNA1C" },
  { name: "hydrochlorothiazide", drugClass: "thiazide_diuretic", geneSymbol: "SLC12A3" },
  { name: "chlorthalidone", drugClass: "thiazide_diuretic", geneSymbol: "SLC12A3" },
  { name: "indapamide", drugClass: "thiazide_diuretic", geneSymbol: "SLC12A3" },
];

export const DRUG_CLASS_LABELS: Record<DrugClass, string> = {
  ace_inhibitor: "ACE inhibitor",
  arb: "ARB (angiotensin receptor blocker)",
  beta_blocker: "Beta-blocker",
  calcium_channel_blocker: "Calcium-channel blocker",
  thiazide_diuretic: "Thiazide diuretic",
  unknown: "Unknown",
};

export const COMORBIDITY_LABELS: Record<string, string> = {
  type2_diabetes: "Type 2 diabetes",
  chronic_kidney_disease: "Chronic kidney disease",
  heart_failure: "Heart failure",
  asthma_copd: "Asthma / COPD",
};
