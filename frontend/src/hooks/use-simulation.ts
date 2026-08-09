"use client";

import { useCallback, useRef, useState } from "react";

import { streamSimulation } from "@/lib/api";
import type {
  AgentStatePartial,
  PatientProfile,
  PipelineStage,
} from "@/lib/types";
import { PIPELINE_STAGES } from "@/lib/types";

// "skipped" covers a real graph behavior, not a frontend edge case: the
// backend has two conditional short-circuit branches (see
// app/agent/graph.py) that jump straight to generate_report when entity
// resolution fails or an unsupported combination size is requested —
// e.g. parse_query hitting its known ~20% malformed-JSON failure mode
// leaves 0 resolved compounds, so run_prediction/retrieve_comparators
// never execute at all. Without an explicit "skipped" state, those
// stages were left dangling "active"/"pending" forever once the run
// finished — a permanently-spinning stage and a nonsensical negative
// elapsed time, both found via a real run hitting this exact path.
export type StageStatus = "pending" | "active" | "done" | "skipped";

/** Wall-clock ms timestamps (Date.now()), recorded client-side as each
 * real SSE event arrives — used to show REAL elapsed time per stage,
 * not a fabricated progress percentage (ui_build_brief.md: timing must
 * be honest, not decorative). */
export interface StageTiming {
  startedAt: number | null;
  endedAt: number | null;
}

interface SimulationRunState {
  status: "idle" | "running" | "done" | "error";
  stageStatus: Record<PipelineStage, StageStatus>;
  stageTiming: Record<PipelineStage, StageTiming>;
  state: AgentStatePartial;
  error: string | null;
}

function initialStageStatus(): Record<PipelineStage, StageStatus> {
  return Object.fromEntries(
    PIPELINE_STAGES.map((s) => [s, "pending"])
  ) as Record<PipelineStage, StageStatus>;
}

function initialStageTiming(): Record<PipelineStage, StageTiming> {
  return Object.fromEntries(
    PIPELINE_STAGES.map((s) => [s, { startedAt: null, endedAt: null }])
  ) as Record<PipelineStage, StageTiming>;
}

/** Marks every stage before `upToIndex` that isn't already "done" as
 * "skipped" — real graph short-circuits jump the SSE event sequence
 * forward, and anything left "active"/"pending" at that point genuinely
 * didn't run, not "hasn't run yet". Also clears its timing so no stale
 * startedAt survives to produce a bogus elapsed time later. */
function resolveSkippedStages(
  stageStatus: Record<PipelineStage, StageStatus>,
  stageTiming: Record<PipelineStage, StageTiming>,
  upToIndex: number
): {
  stageStatus: Record<PipelineStage, StageStatus>;
  stageTiming: Record<PipelineStage, StageTiming>;
} {
  const nextStatus = { ...stageStatus };
  const nextTiming = { ...stageTiming };
  for (let i = 0; i < upToIndex; i++) {
    const stage = PIPELINE_STAGES[i];
    if (nextStatus[stage] !== "done") {
      nextStatus[stage] = "skipped";
      nextTiming[stage] = { startedAt: null, endedAt: null };
    }
  }
  return { stageStatus: nextStatus, stageTiming: nextTiming };
}

/**
 * Drives POST /simulate/stream and accumulates the real per-node state
 * updates into one merged AgentState-shaped object, plus a stage-status
 * map the pipeline visualization reads directly (real node-completion
 * events, not a client-side timer — see ui_build_brief.md's "timing
 * needs to be honest, not decorative").
 */
export function useSimulation() {
  const [run, setRun] = useState<SimulationRunState>({
    status: "idle",
    stageStatus: initialStageStatus(),
    stageTiming: initialStageTiming(),
    state: {},
    error: null,
  });
  const abortRef = useRef<AbortController | null>(null);

  const start = useCallback(async (rawQuery: string, patient: PatientProfile) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const startedAt = Date.now();
    setRun({
      status: "running",
      // First stage starts "active" immediately on click, not "pending"
      // — otherwise the UI shows zero feedback for the ~35-40s until
      // parse_query's own SSE event arrives, which is exactly the
      // "looks frozen" failure mode the brief warns against.
      stageStatus: { ...initialStageStatus(), [PIPELINE_STAGES[0]]: "active" },
      stageTiming: {
        ...initialStageTiming(),
        [PIPELINE_STAGES[0]]: { startedAt, endedAt: null },
      },
      state: {},
      error: null,
    });

    try {
      for await (const evt of streamSimulation(rawQuery, patient, controller.signal)) {
        if (evt.event === "error") {
          setRun((prev) => {
            // A mid-run failure — anything still active/pending genuinely
            // didn't complete, not "still to come".
            const swept = resolveSkippedStages(prev.stageStatus, prev.stageTiming, PIPELINE_STAGES.length);
            return { ...prev, status: "error", error: evt.data.message, ...swept };
          });
          return;
        }
        if (evt.event === "done") {
          setRun((prev) => {
            // The graph has two conditional short-circuits straight to
            // generate_report (see app/agent/graph.py) — if the last real
            // stage event wasn't generate_report reached via every prior
            // stage, some stages never ran at all. Sweep them to
            // "skipped" rather than leaving a stage spinning forever.
            const swept = resolveSkippedStages(prev.stageStatus, prev.stageTiming, PIPELINE_STAGES.length);
            return {
              ...prev,
              status: "done",
              ...swept,
              state: { ...prev.state, ...evt.data },
            };
          });
          return;
        }
        // Real pipeline stage event — mark it done, and mark the NEXT
        // stage active so the UI always shows forward progress rather
        // than a gap between "this stage finished" and "the next one's
        // first token arrives".
        const stageIndex = PIPELINE_STAGES.indexOf(evt.event);
        const now = Date.now();
        setRun((prev) => {
          // Any stage strictly before this one that isn't "done" was
          // jumped over by a real conditional short-circuit, not merely
          // "next in line" — resolve those first so the sweep below only
          // has to handle the current + next stage.
          const { stageStatus: sweptStatus, stageTiming: sweptTiming } = resolveSkippedStages(
            prev.stageStatus,
            prev.stageTiming,
            stageIndex
          );
          const nextStageStatus = { ...sweptStatus, [evt.event]: "done" as const };
          const nextStageTiming = {
            ...sweptTiming,
            [evt.event]: {
              // Falls back to `now` for both ends if this stage's event
              // arrived without ever having been marked "active" first
              // (i.e. it was itself jumped to directly) — we genuinely
              // don't know when it started, so ~0s is the honest answer,
              // not a guessed duration.
              startedAt: prev.stageTiming[evt.event].startedAt ?? now,
              endedAt: now,
            },
          };
          const next = PIPELINE_STAGES[stageIndex + 1];
          if (next) {
            nextStageStatus[next] = "active";
            nextStageTiming[next] = { startedAt: now, endedAt: null };
          }
          return {
            ...prev,
            stageStatus: nextStageStatus,
            stageTiming: nextStageTiming,
            state: { ...prev.state, ...evt.data },
          };
        });
      }
    } catch (err) {
      if (controller.signal.aborted) return;
      setRun((prev) => ({
        ...prev,
        status: "error",
        error: err instanceof Error ? err.message : "Simulation failed.",
      }));
    }
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setRun({
      status: "idle",
      stageStatus: initialStageStatus(),
      stageTiming: initialStageTiming(),
      state: {},
      error: null,
    });
  }, []);

  return { ...run, start, reset };
}
