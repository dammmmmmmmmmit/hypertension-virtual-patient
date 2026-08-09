"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { Check, Loader2, Minus } from "lucide-react";

import { PIPELINE_STAGES, type PipelineStage, type TokenUsage } from "@/lib/types";
import type { StageStatus, StageTiming } from "@/hooks/use-simulation";

const STAGE_LABELS: Record<PipelineStage, string> = {
  parse_query: "Reading your query",
  resolve_entities: "Looking up drug data",
  retrieve_data: "Retrieving mechanism context",
  structure_features: "Structuring features",
  run_prediction: "Running the prediction model",
  retrieve_comparators: "Checking standard-of-care comparisons",
  generate_report: "Writing your report",
};

// Real expected durations (README / DECISIONS.md #7) — used only to
// decide which stages get rotating sub-step copy (the ones long enough
// that a static label would look frozen), never to fake a percentage.
const EXPECTED_SECONDS: Partial<Record<PipelineStage, string>> = {
  parse_query: "~35–40s",
  generate_report: "~90–130s",
};

// Illustrative likely-activity text for the two slow, LLM-calling
// stages — NOT a live trace (the backend emits one SSE event per whole
// node completion, no finer-grained telemetry exists). Grounded in real
// implementation details (4-bit load, the mandated 5-section report
// structure, the known retry-on-malformed-JSON path) so it's honest
// about what's *likely* happening, not fabricated busywork text.
const SUB_STEPS: Partial<Record<PipelineStage, string[]>> = {
  parse_query: [
    "Loading fine-tuned model weights (4-bit)…",
    "Running inference on your query…",
    "Validating structured output against the schema…",
    "Retrying on malformed output, if needed…",
  ],
  generate_report: [
    "Loading qwen2.5:14b-instruct…",
    "Drafting the mechanism summary…",
    "Writing the predicted efficacy…",
    "Writing the side-effect profile…",
    "Comparing against standard-of-care…",
    "Compiling caveats…",
  ],
};

function elapsedLabel(timing: StageTiming, now: number): string {
  if (!timing.startedAt) return "";
  const end = timing.endedAt ?? now;
  const s = (end - timing.startedAt) / 1000;
  return s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

function TokenUsageLine({ usage }: { usage: TokenUsage }) {
  return (
    <p className="mt-0.5 truncate text-xs text-muted-foreground/80">
      <span className="text-foreground/70">{usage.model}</span>
      {" — "}
      <span className="font-mono">
        {usage.input_tokens.toLocaleString()} in / {usage.output_tokens.toLocaleString()} out
        {usage.attempts > 1 && ` (${usage.attempts} attempts)`}
      </span>
    </p>
  );
}

// Keyed by the stage's real start timestamp in the parent (see below) so
// each activation gets a fresh mount — cycling always starts from the
// first sub-step without needing an explicit reset inside an effect.
function SubStepText({
  stage,
  active,
  reduceMotion,
}: {
  stage: PipelineStage;
  active: boolean;
  reduceMotion: boolean;
}) {
  const steps = SUB_STEPS[stage];
  const [i, setI] = useState(0);

  useEffect(() => {
    if (!active || !steps) return;
    const id = setInterval(() => setI((n) => (n + 1) % steps.length), 4500);
    return () => clearInterval(id);
  }, [active, steps]);

  if (!active || !steps) return null;

  // The rotating text itself is real information (not decoration), so
  // reduced-motion keeps it — only the slide/fade transition between
  // entries is dropped in favor of an instant swap.
  if (reduceMotion) {
    return <p className="text-xs text-muted-foreground">{steps[i]}</p>;
  }

  return (
    <AnimatePresence mode="wait">
      <motion.p
        key={i}
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -4 }}
        transition={{ duration: 0.25 }}
        className="text-xs text-muted-foreground"
      >
        {steps[i]}
      </motion.p>
    </AnimatePresence>
  );
}

interface PipelineVisualizationProps {
  stageStatus: Record<PipelineStage, StageStatus>;
  stageTiming: Record<PipelineStage, StageTiming>;
  tokenUsage?: Partial<Record<PipelineStage, TokenUsage>>;
  running: boolean;
}

/**
 * The signature pipeline visualization — wired to real SSE events via
 * useSimulation, not a client-side fake timer. Elapsed times shown per
 * stage are real Date.now() deltas recorded as events actually arrive.
 * Respects prefers-reduced-motion: the pulsing/looping animations are
 * dropped in favor of instant, static state changes.
 */
export function PipelineVisualization({
  stageStatus,
  stageTiming,
  tokenUsage,
  running,
}: PipelineVisualizationProps) {
  const reduceMotion = useReducedMotion();
  const [now, setNow] = useState(() => Date.now());
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!running) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      return;
    }
    intervalRef.current = setInterval(() => setNow(Date.now()), 1000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [running]);

  const doneCount = PIPELINE_STAGES.filter((s) => stageStatus[s] === "done").length;
  // A real run can reach a terminal state without every stage running —
  // the graph short-circuits straight to generate_report when parsing
  // or entity resolution fails (see useSimulation's resolveSkippedStages).
  // Once nothing is left active/pending, the run is over regardless of
  // how many stages were skipped, so the rail should read as complete
  // rather than stalling partway because doneCount alone undercounts it.
  const isTerminal = PIPELINE_STAGES.every(
    (s) => stageStatus[s] === "done" || stageStatus[s] === "skipped"
  );
  const railFraction = isTerminal
    ? 1
    : Math.min(doneCount, PIPELINE_STAGES.length - 1) / (PIPELINE_STAGES.length - 1);

  return (
    <div className="rounded-lg border border-border bg-card p-4 sm:p-5">
      <div className="relative flex flex-col">
        {/* Background rail + fill showing real completed-stage progress */}
        <div className="absolute top-3 bottom-3 left-[13px] w-px bg-border" aria-hidden />
        <motion.div
          className="absolute top-3 left-[13px] w-px origin-top bg-acid"
          initial={false}
          animate={{
            height: `calc(${railFraction * 100}% - ${doneCount === 0 && !isTerminal ? 0 : 12}px)`,
          }}
          transition={reduceMotion ? { duration: 0 } : { duration: 0.4, ease: "easeOut" }}
          aria-hidden
        />

        {PIPELINE_STAGES.map((stage, i) => {
          const status = stageStatus[stage];
          const timing = stageTiming[stage];
          const isActive = status === "active";
          const isDone = status === "done";
          const isSkipped = status === "skipped";

          return (
            <div key={stage} className={`relative flex gap-3 ${i < PIPELINE_STAGES.length - 1 ? "pb-5" : ""}`}>
              <div className="relative z-10 flex size-7 shrink-0 items-center justify-center rounded-full border bg-card">
                {isDone && (
                  <motion.div
                    initial={reduceMotion ? false : { scale: 0.6, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="flex size-full items-center justify-center rounded-full border-acid bg-acid/15"
                  >
                    <Check className="size-3.5 text-acid" />
                  </motion.div>
                )}
                {isActive && (
                  <motion.div
                    className="flex size-full items-center justify-center rounded-full border-acid"
                    animate={
                      reduceMotion
                        ? {}
                        : { boxShadow: ["0 0 0 0 rgba(204,255,0,0.35)", "0 0 0 6px rgba(204,255,0,0)"] }
                    }
                    transition={reduceMotion ? undefined : { duration: 1.4, repeat: Infinity }}
                  >
                    <Loader2 className="size-3.5 animate-spin text-acid" />
                  </motion.div>
                )}
                {status === "pending" && (
                  <div className="size-full rounded-full border border-border" />
                )}
                {isSkipped && (
                  <div className="flex size-full items-center justify-center rounded-full border border-border bg-muted/40">
                    <Minus className="size-3 text-muted-foreground" />
                  </div>
                )}
              </div>

              <div className="flex-1 pt-0.5">
                <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5">
                  <p
                    className={`text-sm font-medium ${
                      status === "pending" || isSkipped
                        ? "text-muted-foreground"
                        : "text-foreground"
                    }`}
                  >
                    {STAGE_LABELS[stage]}
                  </p>
                  <span className="font-mono text-xs text-muted-foreground">
                    {isDone && elapsedLabel(timing, now)}
                    {isActive && (
                      <span className="text-acid">{elapsedLabel(timing, now)}</span>
                    )}
                    {status === "pending" && EXPECTED_SECONDS[stage]}
                    {isSkipped && "not run"}
                  </span>
                </div>
                {isSkipped ? (
                  <p className="mt-0.5 h-4 text-xs text-muted-foreground/70">
                    Skipped — the pipeline short-circuited to the report
                    before reaching this stage.
                  </p>
                ) : (
                  <div className="mt-0.5 h-4">
                    <SubStepText
                      key={timing.startedAt ?? stage}
                      stage={stage}
                      active={isActive}
                      reduceMotion={!!reduceMotion}
                    />
                  </div>
                )}
                {/* Only parse_query and generate_report ever call an
                    LLM — every other stage's tokenUsage[stage] is
                    undefined, so this renders nowhere else. */}
                {isDone && tokenUsage?.[stage] && <TokenUsageLine usage={tokenUsage[stage]} />}
              </div>
            </div>
          );
        })}
      </div>

      {tokenUsage && Object.keys(tokenUsage).length > 0 && (
        <TokenUsageTotal tokenUsage={tokenUsage} />
      )}

      <p className="mt-2 text-[11px] text-muted-foreground/70">
        Sub-step text is illustrative of typical activity during that stage,
        not a live trace — the pipeline reports one event per completed
        stage, not finer-grained progress.
      </p>
    </div>
  );
}

/** Sums whatever LLM-calling stages actually ran — 1 or 2 entries
 * depending on whether the run reached generate_report's real LLM call
 * or stopped at a clarification short-circuit (see parse_query.py /
 * generate_report.py — no LLM call means no entry, not a zero one). */
function TokenUsageTotal({
  tokenUsage,
}: {
  tokenUsage: Partial<Record<PipelineStage, TokenUsage>>;
}) {
  const entries = Object.values(tokenUsage).filter((u): u is TokenUsage => !!u);
  const total = entries.reduce((sum, u) => sum + u.total_tokens, 0);

  return (
    <div className="mt-3 flex flex-wrap items-center justify-between gap-x-4 gap-y-1 rounded-md border border-indigo/20 bg-indigo/5 px-3 py-2 text-xs">
      <span className="text-muted-foreground">
        {entries.map((u) => u.model).join(" + ")}
      </span>
      <span className="font-mono font-medium text-foreground">
        {total.toLocaleString()} tokens total
      </span>
    </div>
  );
}
