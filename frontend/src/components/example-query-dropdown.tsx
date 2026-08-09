"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { EXAMPLE_QUERIES } from "@/lib/example-queries";

interface ExampleQueryDropdownProps {
  onSelect: (query: string) => void;
  disabled?: boolean;
}

/**
 * Replaces the old 3 inline example-question chips — same job (populate
 * the query box, still editable after selection), just information-dense
 * enough to hold the full realistic range instead of 3 cherry-picked
 * ones. Deliberately restrained visually (a plain Select, not a
 * headline feature) per the redesign brief — the page's one
 * signature-animation budget goes to the pipeline visualization, not
 * here.
 *
 * The one color touch: once a query is chosen, the trigger's displayed
 * text renders in acid — the same "mark the thing currently driving the
 * interaction" logic as the Run button, extended to this one spot per
 * the brief's explicit color-discipline note. Every other dropdown on
 * this page (age/sex, etc.) stays neutral.
 */
export function ExampleQueryDropdown({ onSelect, disabled }: ExampleQueryDropdownProps) {
  return (
    <Select
      onValueChange={(value) => {
        if (typeof value === "string") onSelect(value);
      }}
      disabled={disabled}
    >
      <SelectTrigger className="w-full [&_[data-slot=select-value]:not([data-placeholder])]:text-acid">
        <SelectValue placeholder="Or pick an example query…" />
      </SelectTrigger>
      <SelectContent className="max-w-[min(28rem,90vw)]">
        {EXAMPLE_QUERIES.map((q) => (
          <SelectItem
            key={q.id}
            value={q.text}
            className="whitespace-normal focus:bg-acid/15 focus:text-foreground"
          >
            {q.text}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
