import type { PatientProfile, SimulationEvent } from "./types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

/**
 * Streams the real agent pipeline (app/agent/graph.py, via
 * app/api/main.py's POST /simulate/stream) as it actually runs, one
 * event per real LangGraph node completion.
 *
 * Uses `fetch()` + manual SSE parsing rather than the browser's native
 * `EventSource` — EventSource only supports GET requests, and this
 * endpoint needs a JSON body (query + patient profile). This is the
 * standard pattern for "SSE over POST".
 */
export async function* streamSimulation(
  rawQuery: string,
  patient: PatientProfile,
  signal?: AbortSignal
): AsyncGenerator<SimulationEvent> {
  const response = await fetch(`${API_BASE_URL}/simulate/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ raw_query: rawQuery, patient }),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(
      `Simulation request failed: ${response.status} ${response.statusText}`
    );
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line ("\n\n").
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);

      const eventLine = frame.split("\n").find((l) => l.startsWith("event: "));
      const dataLine = frame.split("\n").find((l) => l.startsWith("data: "));
      if (eventLine && dataLine) {
        const event = eventLine.slice("event: ".length).trim();
        const data = JSON.parse(dataLine.slice("data: ".length));
        yield { event, data } as SimulationEvent;
      }

      boundary = buffer.indexOf("\n\n");
    }
  }
}
