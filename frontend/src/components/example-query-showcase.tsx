"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { SHOWCASE_QUERIES } from "@/lib/example-queries";

interface ExampleQueryShowcaseProps {
  onSelect: (query: string) => void;
  disabled?: boolean;
}

/**
 * A curated subset of EXAMPLE_QUERIES (same source as the dropdown, see
 * lib/example-queries.ts) shown as clickable cards rather than list
 * items — picked to span a plain question, a combo, and the two
 * deliberately-flagged edge cases (discouraged combo, thiazide data
 * gap). Hover state is indigo, not acid — clicking one of these sets up
 * a run, it isn't the run itself.
 */
export function ExampleQueryShowcase({ onSelect, disabled }: ExampleQueryShowcaseProps) {
  return (
    <Card className="border-indigo/25">
      <CardHeader>
        <CardTitle className="text-base">Worth trying</CardTitle>
        <p className="text-xs text-muted-foreground">
          A few queries that show the range, including two flagged edge cases.
        </p>
      </CardHeader>
      <CardContent className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {SHOWCASE_QUERIES.map((q) => (
          <button
            key={q.id}
            type="button"
            disabled={disabled}
            onClick={() => onSelect(q.text)}
            className="rounded-lg border border-indigo/20 bg-indigo/5 p-3 text-left text-xs text-foreground/90 transition-colors hover:border-indigo/50 hover:bg-indigo/10 disabled:pointer-events-none disabled:opacity-50"
          >
            {q.tag && (
              <Badge variant="outline" className="mb-1.5 border-indigo/30 text-indigo text-[10px]">
                {q.tag}
              </Badge>
            )}
            <p>{q.text}</p>
          </button>
        ))}
      </CardContent>
    </Card>
  );
}
