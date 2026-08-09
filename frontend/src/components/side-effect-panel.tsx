import { AlertTriangle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface SideEffectPanelProps {
  probabilities: Record<string, number>;
  /** How many resolved compounds this prediction covers — drives the
   * evidence label (README: combo side-effect probabilities are an
   * element-wise max across components, not a learned interaction). */
  compoundCount: number;
}

type Tier = "top" | "mid" | "low";

const TIER_LABEL: Record<Tier, string> = {
  top: "Top-ranked",
  mid: "Mid-ranked",
  low: "Lower-ranked",
};

const TIER_BADGE_CLASS: Record<Tier, string> = {
  top: "bg-acid/15 text-acid border-acid/30",
  mid: "bg-muted text-foreground border-border",
  low: "bg-muted text-muted-foreground border-border",
};

function tierFor(index: number, n: number): Tier {
  if (index < Math.ceil(n / 3)) return "top";
  if (index < Math.ceil((2 * n) / 3)) return "mid";
  return "low";
}

/**
 * Structured side-effect display, rendered from the real
 * side_effect_probabilities dict directly — NOT from LLM prose — so the
 * honesty framing here doesn't depend on the report-writer model
 * phrasing it correctly on every run.
 *
 * The underlying LightGBM model (14 training rows, 30 labels — see
 * DECISIONS.md #4 and the data & methodology page) produces raw
 * probabilities that are consistently near-saturated (~0.99+) regardless
 * of drug. Showing those as percentages or a magnitude-scaled bar would
 * manufacture a precision the model doesn't have. What IS real signal,
 * per the same documented finding, is RANK — the top-predicted labels do
 * correctly differ by drug even though their absolute values don't.
 *
 * So every visual here is rank-derived, not probability-magnitude-derived:
 * - "Confidence band" = which third of the returned ranking this entry
 *   falls in, not a calibrated confidence interval.
 * - The bar width = ordinal rank position, not the raw probability — a
 *   magnitude-scaled bar across values like 0.9990-0.9993 would either
 *   render as 8 identical full bars or require artificial rescaling that
 *   invents a difference that isn't really there.
 * - "Evidence" = whether this is a single-compound estimate or a
 *   combination (element-wise max across components, not a learned
 *   interaction) — a real, disclosed fact about how the number was
 *   built, not a per-row data-quality score we don't actually have.
 *
 * Note on naming: this deliberately says "relative rank," not "relative
 * risk" — relative risk is a specific epidemiological ratio (probability
 * in exposed vs. unexposed) this project has no baseline data to
 * compute. Labeling an ordinal rank bar as "relative risk" would be
 * exactly the kind of overstated claim this project's design log
 * (DECISIONS.md) exists to prevent.
 */
export function SideEffectPanel({ probabilities, compoundCount }: SideEffectPanelProps) {
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) return null;

  const n = entries.length;
  const evidenceLabel =
    compoundCount > 1 ? "Combination — max across components" : "Single-compound estimate";

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center justify-between gap-2 text-base">
          Predicted side effects
          <Badge variant="outline" className="font-mono text-[10px]">
            {evidenceLabel}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-start gap-2 rounded-md border border-caveat/30 bg-caveat/10 px-3 py-2 text-xs text-caveat">
          <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
          <p>
            This model&rsquo;s raw probabilities are near-saturated (~0.99+
            for almost every label) — an artifact of training on only 14
            drugs. Treat the <span className="font-medium">ranking</span>{" "}
            below as the signal; the raw values (shown small, for
            transparency) are not calibrated likelihoods.
          </p>
        </div>

        <ul className="space-y-2">
          {entries.map(([name, p], i) => {
            const tier = tierFor(i, n);
            const rankPct = ((n - i) / n) * 100;
            return (
              <li key={name} className="space-y-1">
                <div className="flex items-center justify-between gap-2 text-sm">
                  <span className="flex items-center gap-2">
                    <span className="font-mono text-xs text-muted-foreground">#{i + 1}</span>
                    <span className="text-foreground">{name}</span>
                  </span>
                  <span className="flex items-center gap-2">
                    <Badge variant="outline" className={`text-[10px] ${TIER_BADGE_CLASS[tier]}`}>
                      {TIER_LABEL[tier]}
                    </Badge>
                    <span className="font-mono text-[10px] text-muted-foreground/70">
                      raw {p.toFixed(4)}
                    </span>
                  </span>
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-acid/70"
                    style={{ width: `${rankPct}%` }}
                    aria-hidden
                  />
                </div>
              </li>
            );
          })}
        </ul>
        <p className="text-[11px] text-muted-foreground/70">
          Bar length reflects relative rank within this list, not the raw
          probability magnitude — see the data &amp; methodology page for
          why.
        </p>
      </CardContent>
    </Card>
  );
}
