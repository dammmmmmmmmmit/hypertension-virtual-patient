"use client";

import { useState } from "react";
import { Loader2 } from "lucide-react";

import { DrugRegistryPanel } from "@/components/drug-registry-panel";
import { ExampleQueryDropdown } from "@/components/example-query-dropdown";
import { ExampleQueryShowcase } from "@/components/example-query-showcase";
import { PatientForm, DEFAULT_PATIENT } from "@/components/patient-form";
import { PipelineVisualization } from "@/components/pipeline-visualization";
import { QuickStatsPanel } from "@/components/quick-stats-panel";
import { ReportDisplay } from "@/components/report-display";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { useSimulation } from "@/hooks/use-simulation";

export default function Home() {
  const [rawQuery, setRawQuery] = useState("");
  const [patient, setPatient] = useState(DEFAULT_PATIENT);
  const { status, stageStatus, stageTiming, state, error, start, reset } = useSimulation();

  const isRunning = status === "running";

  function handleRun() {
    if (!rawQuery.trim() || isRunning) return;
    start(rawQuery, patient);
  }

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <header className="mb-8 space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">
          Virtual Patient Drug-Response Simulator
        </h1>
        <p className="text-sm font-medium text-muted-foreground">
          An LLM-powered Clinical Decision Support System for Hypertension
          Medication Analysis
        </p>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Describe a hypertension medication question in plain language. The
          real pipeline resolves the compounds against ChEMBL/PubChem/SIDER,
          runs the prediction model, and drafts a clinician-readable report —
          a screening aid for researchers, not a point-of-care tool.
        </p>
      </header>

      {/* Bento dashboard — row 1: query+patient (larger) | drug registry.
          items-start so the shorter registry panel doesn't get stretched
          to match the taller query+patient column (it scrolls internally
          instead — see DrugRegistryPanel's max-h). */}
      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[1.6fr_1fr]">
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Your query</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Textarea
                value={rawQuery}
                onChange={(e) => setRawQuery(e.target.value)}
                placeholder="e.g. How would adding 50mg losartan affect this patient's blood pressure?"
                rows={4}
                disabled={isRunning}
              />
              <ExampleQueryDropdown onSelect={setRawQuery} disabled={isRunning} />
              <div className="flex items-center gap-3 pt-1">
                <Button
                  variant="accent"
                  onClick={handleRun}
                  disabled={isRunning || !rawQuery.trim()}
                >
                  {isRunning && <Loader2 className="size-4 animate-spin" />}
                  {isRunning ? "Running simulation…" : "Run simulation"}
                </Button>
                {status !== "idle" && !isRunning && (
                  <button
                    type="button"
                    onClick={reset}
                    className="text-sm text-muted-foreground hover:text-foreground"
                  >
                    Reset
                  </button>
                )}
              </div>
              {isRunning && (
                <p className="text-xs text-muted-foreground">
                  Full runs typically take 2–3 minutes — the parsing and
                  report-writing stages are the slowest (~35–40s and
                  ~90–130s respectively), since both call a local language
                  model.
                </p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Patient profile</CardTitle>
            </CardHeader>
            <CardContent>
              <PatientForm value={patient} onChange={setPatient} />
            </CardContent>
          </Card>
        </div>

        <DrugRegistryPanel onSelectDrug={setRawQuery} disabled={isRunning} />
      </div>

      {/* Bento row 2: quick stats | example query showcase */}
      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <QuickStatsPanel />
        <ExampleQueryShowcase onSelect={setRawQuery} disabled={isRunning} />
      </div>

      {/* Results — full width, breaks out of the bento grid once a run
          starts, since pipeline/report height is inherently variable. */}
      {(isRunning || status === "done" || status === "error") && (
        <div className="mt-6 space-y-4">
          <PipelineVisualization
            stageStatus={stageStatus}
            stageTiming={stageTiming}
            tokenUsage={state.token_usage}
            running={isRunning}
          />

          {status === "error" && (
            <Card className="border-magenta/40">
              <CardContent className="pt-6 text-sm text-magenta">
                {error ?? "Something went wrong running the simulation."}
              </CardContent>
            </Card>
          )}

          {status === "done" && (
            <ReportDisplay
              report={state.report ?? ""}
              discouragedWarning={state.discouraged_warning ?? null}
              prediction={state.prediction ?? null}
              compounds={state.compounds ?? []}
            />
          )}
        </div>
      )}
    </div>
  );
}
