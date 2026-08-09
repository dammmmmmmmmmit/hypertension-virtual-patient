import {
  AlertTriangle,
  FlaskConical,
  Pill,
  Scale,
  ShieldAlert,
  Sparkles,
} from "lucide-react";
import ReactMarkdown from "react-markdown";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { SideEffectPanel } from "@/components/side-effect-panel";
import { parseReportSections, type ReportSection } from "@/lib/parse-report";
import type { CompoundQuery, DiseaseParameterDelta, Prediction } from "@/lib/types";

const SECTION_ICON: Record<ReportSection["key"], React.ComponentType<{ className?: string }>> = {
  mechanism: FlaskConical,
  efficacy: Sparkles,
  // Rendered separately by SideEffectPanel below, from the real
  // probability dict — this entry stays only so the Record type below
  // stays exhaustive; the LLM's own prose "side_effects" section is
  // filtered out of the render (see the sections.filter below).
  side_effects: Pill,
  comparison: Scale,
  caveats: AlertTriangle,
  other: FlaskConical,
};

function DeltaStat({ delta }: { delta: DiseaseParameterDelta }) {
  const label = delta.parameter === "systolic_bp" ? "Systolic Δ" : "Diastolic Δ";
  return (
    <div className="rounded-lg border border-border bg-card px-4 py-3">
      <p className="text-xs uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="font-mono text-2xl font-semibold text-acid">
        {delta.predicted_delta > 0 ? "+" : ""}
        {delta.predicted_delta.toFixed(1)}
        <span className="ml-1 text-sm text-muted-foreground">mmHg</span>
      </p>
      <p className="font-mono text-xs text-muted-foreground">
        {delta.baseline_value.toFixed(0)} → {delta.predicted_value.toFixed(1)} mmHg
      </p>
    </div>
  );
}

interface ReportDisplayProps {
  report: string;
  discouragedWarning: string | null;
  prediction: Prediction | null;
  compounds: CompoundQuery[];
}

/**
 * Renders the 5-section report (see app/agent/generate_report.py /
 * DECISIONS.md #7's SYSTEM_PROMPT — mechanism, efficacy, side effects,
 * comparison, caveats, every time). The discouraged-combination warning
 * renders as its own unmistakable magenta banner ABOVE the report body —
 * never buried in prose (ui_build_brief.md, DISCOURAGED_COMBINATIONS).
 *
 * The LLM's own "side effects" prose section is deliberately dropped in
 * favor of SideEffectPanel, which reads prediction.side_effect_probabilities
 * directly — see that component's docstring for why (the underlying model's
 * raw probabilities are near-saturated and shouldn't be phrased as
 * percentages, which is easy for prose to get wrong run to run even with a
 * corrected prompt).
 */
export function ReportDisplay({
  report,
  discouragedWarning,
  prediction,
  compounds,
}: ReportDisplayProps) {
  const sections = parseReportSections(report).filter((s) => s.key !== "side_effects");

  return (
    <div className="space-y-4">
      {discouragedWarning && (
        <Alert variant="discouraged">
          <ShieldAlert />
          <AlertTitle>Discouraged combination</AlertTitle>
          <AlertDescription className="text-white/90">
            {discouragedWarning}
          </AlertDescription>
        </Alert>
      )}

      {prediction && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {prediction.deltas.map((d) => (
            <DeltaStat key={d.parameter} delta={d} />
          ))}
        </div>
      )}

      {prediction && (
        <SideEffectPanel
          probabilities={prediction.side_effect_probabilities}
          compoundCount={compounds.length}
        />
      )}

      {sections.map((section, i) => {
        const Icon = SECTION_ICON[section.key];
        const isCaveats = section.key === "caveats";
        return (
          <Card
            key={i}
            className={isCaveats ? "border-caveat/30" : undefined}
          >
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Icon
                  className={`size-4 ${isCaveats ? "text-caveat" : "text-acid"}`}
                />
                {section.title}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="prose prose-invert prose-sm max-w-none prose-p:text-foreground/90 prose-li:text-foreground/90 prose-headings:text-foreground">
                <ReactMarkdown>{section.body}</ReactMarkdown>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
