/**
 * TypeScript mirror of app/agent/state.py's AgentState + the Pydantic
 * schemas it holds (app/schemas/patient.py, app/schemas/compound.py).
 * Kept as a hand-written mirror, not generated, because the backend
 * exposes this shape via SSE state-diffs (app/api/main.py's
 * /simulate/stream), not a typed OpenAPI response body. Field names
 * match exactly — verified against a live curl test of the real
 * endpoint (see chat history), not assumed from the Python source alone.
 */

export type Sex = "male" | "female";

export type Comorbidity =
  | "none"
  | "type2_diabetes"
  | "chronic_kidney_disease"
  | "heart_failure"
  | "asthma_copd";

export type DrugClass =
  | "ace_inhibitor"
  | "arb"
  | "beta_blocker"
  | "calcium_channel_blocker"
  | "thiazide_diuretic"
  | "unknown";

export interface DiseaseParameters {
  systolic_bp: number;
  diastolic_bp: number;
  heart_rate: number;
  serum_potassium: number;
  egfr: number;
}

export interface PatientProfile {
  age: number;
  sex: Sex;
  weight_kg: number;
  baseline: DiseaseParameters;
  comorbidities: Comorbidity[];
  current_medications: string[];
}

export interface CompoundQuery {
  raw_name: string;
  resolved_name: string | null;
  drug_class: DrugClass | null;
  chembl_id: string | null;
  pubchem_cid: number | null;
  dose_mg: number | null;
  notes: string | null;
}

export interface ParsedPatient {
  patient: PatientProfile;
  defaulted_fields: string[];
}

export interface MechanismContext {
  drug_name: string;
  mechanism_text: string;
  source: "chembl_mechanism" | "class_level_fallback";
}

export interface DiseaseParameterDelta {
  parameter: string;
  baseline_value: number;
  predicted_value: number;
  predicted_delta: number;
  confidence: number | null;
}

export interface Prediction {
  deltas: DiseaseParameterDelta[];
  raw_model_output_mmHg: { systolic: number; diastolic: number };
  renal_adjustment_factor: number;
  confidence: number;
  side_effect_probabilities: Record<string, number>;
}

export interface ComparatorInfo {
  description: string;
  drug_names: string[];
}

/** Only parse_query and generate_report ever produce an entry — the
 * only two nodes that call an LLM (see app/agent/state.py). `attempts`
 * is meaningful only for parse_query, which has an internal retry loop
 * on its known ~20% malformed-JSON failure mode; input/output/total are
 * already summed across every attempt, successful or not. */
export interface TokenUsage {
  model: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  attempts: number;
}

/** Partial state — each SSE event only carries the FIELDS that node
 * updated, per LangGraph's `.astream()` "updates" mode. Consumers must
 * merge events into an accumulated state, not treat any single event as
 * complete (see useSimulation hook). */
export interface AgentStatePartial {
  raw_query?: string;
  compounds?: CompoundQuery[];
  parsed_patient?: ParsedPatient;
  question_intent?: string | null;
  all_resolved?: boolean;
  unresolved_names?: string[];
  mechanism_contexts?: MechanismContext[];
  prediction?: Prediction | null;
  comparators?: ComparatorInfo[];
  discouraged_warning?: string | null;
  report?: string | null;
  limitations?: string[];
  token_usage?: Partial<Record<PipelineStage, TokenUsage>>;
}

/** The exact 7 real stages, in real order — pulled from app/agent/graph.py,
 * not assumed. "done" and "error" are synthetic events this frontend's
 * SSE consumer adds on top, not real graph nodes. */
export const PIPELINE_STAGES = [
  "parse_query",
  "resolve_entities",
  "retrieve_data",
  "structure_features",
  "run_prediction",
  "retrieve_comparators",
  "generate_report",
] as const;

export type PipelineStage = (typeof PIPELINE_STAGES)[number];

export type SimulationEvent =
  | { event: PipelineStage; data: AgentStatePartial }
  | { event: "done"; data: AgentStatePartial }
  | { event: "error"; data: { message: string } };
