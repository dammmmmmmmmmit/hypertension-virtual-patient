export interface ReportSection {
  key: "mechanism" | "efficacy" | "side_effects" | "comparison" | "caveats" | "other";
  title: string;
  body: string;
}

/**
 * Splits the LLM-generated report (app/agent/generate_report.py's
 * SYSTEM_PROMPT mandates exactly 5 sections in order, but does NOT
 * mandate a markdown heading level) into individually renderable
 * sections.
 *
 * Matches "##" or "###" headings — live testing showed the model
 * mixing levels within a single report (e.g. "### DISCOURAGED..."
 * followed by "## Compound(s)..." for the rest), so splitting on only
 * one level silently swallows every later section into the first.
 *
 * Matches section identity by keyword, not exact header string: the
 * report is real LLM output, not a template, so exact wording can vary
 * run to run even at low temperature. Falls back to "other" for
 * anything unrecognized rather than dropping content — a
 * slightly-differently-worded section heading should still render, not
 * silently vanish.
 */
export function parseReportSections(report: string): ReportSection[] {
  const parts = report.split(/^#{2,3}\s+/m).filter(Boolean);

  return parts
    .map((part) => {
      const [titleLine, ...rest] = part.split("\n");
      const title = titleLine.trim();
      const body = rest.join("\n").trim();
      const lower = title.toLowerCase();

      let key: ReportSection["key"] = "other";
      if (lower.includes("mechanism") || lower.includes("compound")) key = "mechanism";
      else if (lower.includes("efficacy")) key = "efficacy";
      else if (lower.includes("side") || lower.includes("side-effect")) key = "side_effects";
      else if (lower.includes("comparison") || lower.includes("standard")) key = "comparison";
      else if (lower.includes("caveat") || lower.includes("limitation")) key = "caveats";

      return { key, title, body, isDiscouragedHeader: lower.includes("discouraged") || lower.includes("warning") };
    })
    .filter((s) => !s.isDiscouragedHeader)
    .map(({ isDiscouragedHeader: _isDiscouragedHeader, ...s }) => s);
}
