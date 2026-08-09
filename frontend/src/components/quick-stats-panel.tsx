import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

// Real, already-verified numbers (DECISIONS.md — see #7 for the
// fine-tuning before/after figures). Nothing here is rounded or
// invented for the sake of filling a stat tile.
const STATS: { label: string; value: string; sub?: string }[] = [
  { label: "Registry", value: "15", sub: "compounds, 5 drug classes" },
  { label: "Data sources", value: "3", sub: "ChEMBL · PubChem · SIDER" },
  { label: "Fine-tune schema-valid", value: "43% → 80%", sub: "parse_query, before/after QLoRA" },
  { label: "Fine-tune exact-match", value: "30% → 63%", sub: "parse_query, before/after QLoRA" },
];

/**
 * Dashboard stat tiles — real numbers pulled from DECISIONS.md, not
 * decoration. Structural panel, indigo-bordered like the drug registry
 * panel; the numbers themselves render in muted mono, not acid — acid
 * stays reserved for the one active interactive element per view.
 */
export function QuickStatsPanel() {
  return (
    <Card className="border-indigo/25">
      <CardHeader>
        <CardTitle className="text-base">At a glance</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-2 gap-3">
        {STATS.map((s) => (
          <div key={s.label} className="rounded-md border border-indigo/15 bg-indigo/5 px-3 py-2.5">
            <p className="text-[11px] uppercase tracking-wide text-muted-foreground">{s.label}</p>
            <p className="font-mono text-lg font-semibold text-foreground">{s.value}</p>
            {s.sub && <p className="text-[11px] text-muted-foreground">{s.sub}</p>}
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
