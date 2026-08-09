/**
 * Example queries a researcher/grad student would actually type into
 * this tool for early-stage screening work — not generic "try this app"
 * prompts. Includes real edge cases (discouraged combo, thiazide's
 * missing ChEMBL potency data, a misspelled name) worth surfacing
 * deliberately, since the pipeline handles all three honestly and
 * that's worth showing off, not hiding.
 */
export interface ExampleQuery {
  id: string;
  text: string;
  /** Shown only on the showcase cards, not the dropdown — a short tag
   * naming what's notable about this one edge case. */
  tag?: string;
}

export const EXAMPLE_QUERIES: ExampleQuery[] = [
  {
    id: "lisinopril-alone",
    text: "What blood pressure reduction can I expect from lisinopril alone in a 60-year-old with stage 2 hypertension?",
  },
  {
    id: "losartan-asthma",
    text: "Would losartan cause any concerning side effects in a patient with asthma?",
  },
  {
    id: "amlodipine-losartan-combo",
    text: "How does amlodipine plus losartan compare to standard combination therapy?",
  },
  {
    id: "ace-arb-discouraged",
    text: "Is combining an ACE inhibitor with an ARB ever appropriate?",
    tag: "Discouraged-combination flag",
  },
  {
    id: "thiazide-renal",
    text: "What's the expected effect of hydrochlorothiazide in a patient with reduced kidney function?",
    tag: "Thiazide data-gap handling",
  },
  {
    id: "enalapril-thiazide-vs-arb-ccb",
    text: "Compare enalapril plus a thiazide against an ARB plus a calcium-channel blocker for a patient with CKD.",
  },
  {
    id: "metoprolol-heart-failure",
    text: "Would metoprolol be a reasonable addition for a patient with existing heart failure?",
  },
  {
    id: "irbesartan-elderly-renal",
    text: "What are the safety concerns with irbesartan in an elderly patient with declining renal function?",
  },
  {
    id: "vs-guideline",
    text: "How does this compound's predicted efficacy compare to first-line guideline recommendations?",
  },
  {
    id: "misspelled-name",
    text: "What if the patient's history includes a slightly misspelled drug name — does the system catch that?",
    tag: "Fuzzy-match / clarification",
  },
  {
    id: "ramipril-amlodipine",
    text: "Model a 70-year-old male, BP 165/100, on ramipril — what would adding amlodipine do?",
  },
];

/** A smaller curated subset for the showcase-cards bento panel — picked
 * to span the range (a plain question, a combo, and the two flagged
 * edge cases) rather than just the first N in list order. */
export const SHOWCASE_QUERY_IDS = [
  "amlodipine-losartan-combo",
  "ace-arb-discouraged",
  "thiazide-renal",
  "misspelled-name",
];

export const SHOWCASE_QUERIES = SHOWCASE_QUERY_IDS.map(
  (id) => EXAMPLE_QUERIES.find((q) => q.id === id)!
);

export function queryTemplateForDrug(drugName: string): string {
  const label = drugName.charAt(0).toUpperCase() + drugName.slice(1);
  return `How would ${label} affect this patient's blood pressure?`;
}
