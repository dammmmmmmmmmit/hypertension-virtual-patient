"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { DRUG_CLASS_LABELS, HYPERTENSION_DRUGS } from "@/lib/drug-registry";
import { queryTemplateForDrug } from "@/lib/example-queries";
import type { DrugClass } from "@/lib/types";

const CLASS_ORDER: DrugClass[] = [
  "ace_inhibitor",
  "arb",
  "beta_blocker",
  "calcium_channel_blocker",
  "thiazide_diuretic",
];

interface DrugRegistryPanelProps {
  onSelectDrug: (query: string) => void;
  disabled?: boolean;
}

/**
 * Browse the real 15-drug/5-class registry (src/lib/drug-registry.ts,
 * mirrored from app/core/drug_registry.py — see that file for why it's
 * a stable hardcoded list, not a live DB lookup). Functional, not
 * decorative: clicking a drug drops a templated query into the query
 * box. Structural/informational panel, so it carries indigo — not acid,
 * which stays reserved for the run button, the active dropdown
 * selection, and focus rings.
 */
export function DrugRegistryPanel({ onSelectDrug, disabled }: DrugRegistryPanelProps) {
  return (
    <Card className="border-indigo/25">
      <CardHeader>
        <CardTitle className="text-base">Drug registry</CardTitle>
        <p className="text-xs text-muted-foreground">
          15 compounds, 5 classes — click one to start a query.
        </p>
      </CardHeader>
      <CardContent className="max-h-[420px] space-y-4 overflow-y-auto">
        {CLASS_ORDER.map((cls) => (
          <div key={cls}>
            <p className="mb-1.5 text-xs font-medium tracking-wide text-indigo uppercase">
              {DRUG_CLASS_LABELS[cls]}
            </p>
            <div className="flex flex-wrap gap-1.5">
              {HYPERTENSION_DRUGS.filter((d) => d.drugClass === cls).map((d) => (
                <button
                  key={d.name}
                  type="button"
                  disabled={disabled}
                  onClick={() => onSelectDrug(queryTemplateForDrug(d.name))}
                  className="rounded-md border border-indigo/20 bg-indigo/5 px-2.5 py-1 text-xs capitalize text-foreground/90 transition-colors hover:border-indigo/50 hover:bg-indigo/10 disabled:pointer-events-none disabled:opacity-50"
                >
                  {d.name}
                </button>
              ))}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
