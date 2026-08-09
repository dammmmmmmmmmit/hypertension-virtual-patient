import { AlertTriangle, ArrowRight, CheckCircle2, XCircle } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

export const metadata = {
  title: "Architecture & Pipeline — Virtual Patient Drug-Response Simulator",
};

// The real 7 LangGraph nodes (app/agent/graph.py), in real order, with
// real observed timing (see DECISIONS.md #7, README "known limitations").
// Not decorative — these are the numbers the Demo page's pipeline
// visualization is held to.
const STAGES = [
  {
    name: "parse_query",
    summary: "Extract compounds, patient profile, and question intent from raw text.",
    detail:
      "The one fine-tuned node in this project — a QLoRA-tuned Qwen2.5-7B, served via direct in-process 4-bit inference (not Ollama/GGUF, see below).",
    timing: "~35–40s",
  },
  {
    name: "resolve_entities",
    summary: "Match extracted compound names against the ChEMBL/PubChem-backed drug registry.",
    detail: "Cached Postgres lookups, not live API calls — fast.",
    timing: "~sub-second",
  },
  {
    name: "retrieve_data",
    summary: "Pull mechanism-of-action context from the Qdrant RAG index.",
    detail: "FastEmbed BAAI/bge-small-en embeddings over ChEMBL /mechanism.json text.",
    timing: "~sub-second",
  },
  {
    name: "structure_features",
    summary: "Build the feature vector the prediction model expects.",
    detail: "Patient covariates + drug-class one-hots + potency z-score, per DECISIONS.md #1.",
    timing: "~sub-second",
  },
  {
    name: "run_prediction",
    summary: "Run the trained LightGBM efficacy + side-effect models.",
    detail: "Called in-process, not over HTTP — DECISIONS.md #6.",
    timing: "~sub-second",
  },
  {
    name: "retrieve_comparators",
    summary: "Look up standard-of-care comparator combinations for the report.",
    detail: "Static registry lookup.",
    timing: "~sub-second",
  },
  {
    name: "generate_report",
    summary: "Write the clinician-readable report.",
    detail:
      "qwen2.5:14b-instruct via Ollama — deliberately not fine-tuned (no gold-standard report dataset exists).",
    timing: "~90–130s",
  },
];

const BEFORE_AFTER = [
  { metric: "valid_json_rate", before: "97%", after: "80%", afterBetter: false },
  { metric: "schema_valid_rate", before: "43%", after: "80%", afterBetter: true },
  { metric: "exact_field_match_rate", before: "30%", after: "63%", afterBetter: true },
];

export default function ArchitecturePage() {
  return (
    <div className="mx-auto max-w-5xl space-y-12 px-6 py-10">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">Architecture &amp; pipeline</h1>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Seven real LangGraph nodes, one fine-tuned local model, one
          un-fine-tuned local model, and a hard-coded safety rule that
          deliberately isn&rsquo;t a model output. This page documents what was
          actually built, including the parts that didn&rsquo;t go to plan.
        </p>
      </header>

      {/* ---- Pipeline diagram ---- */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">The pipeline (app/agent/graph.py)</h2>
        <p className="text-sm text-muted-foreground">
          Every run passes through all 7 stages in this order, with
          conditional short-circuit branches to a deterministic
          clarification report when entity resolution fails or an
          unsupported combination size is requested — no stage is skipped
          silently.
        </p>
        <div className="flex flex-col gap-3">
          {STAGES.map((stage, i) => (
            <div key={stage.name} className="flex items-start gap-3">
              <div className="flex flex-col items-center pt-1.5">
                <div className="flex size-7 shrink-0 items-center justify-center rounded-full border border-indigo/40 bg-indigo/10 font-mono text-xs text-foreground">
                  {i + 1}
                </div>
                {i < STAGES.length - 1 && (
                  <div className="my-1 h-full w-px flex-1 bg-border" />
                )}
              </div>
              <Card className="mb-1 flex-1 py-4">
                <CardContent className="flex flex-wrap items-start justify-between gap-3 px-4">
                  <div>
                    <p className="font-mono text-sm font-medium text-foreground">
                      {stage.name}
                    </p>
                    <p className="mt-0.5 text-sm text-muted-foreground">{stage.summary}</p>
                    <p className="mt-1 text-xs text-muted-foreground/80">{stage.detail}</p>
                  </div>
                  <Badge variant="outline" className="shrink-0 font-mono">
                    {stage.timing}
                  </Badge>
                </CardContent>
              </Card>
            </div>
          ))}
        </div>
      </section>

      <Separator />

      {/* ---- The pivot story ---- */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Why this runs on local models</h2>
        <p className="text-sm text-muted-foreground">
          The project started on the Anthropic API. Two constraints pointed
          the same direction: no budget for continued API usage, and the
          project&rsquo;s actual goal — demonstrating hands-on LLM engineering
          (attention, fine-tuning, quantization), not just orchestrating a
          hosted API — is better served by open-weight models fine-tuned on
          real hardware (RTX 5070 Ti, 12GB VRAM). Both LLM nodes now run
          entirely locally; there is no Anthropic API dependency left
          anywhere in the app.
        </p>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between text-base">
                generate_report
                <Badge variant="outline">not fine-tuned</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-muted-foreground">
              <p><span className="text-foreground">qwen2.5:14b-instruct</span> via Ollama, inference only.</p>
              <p>
                Deliberately not fine-tuned: open-ended report prose has no
                gold-standard dataset to fine-tune against, and fabricating
                one would be worse than not fine-tuning at all — decided up
                front, not a retroactive excuse.
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between text-base">
                parse_query
                <Badge className="bg-acid/15 text-acid border-acid/30">
                  QLoRA fine-tuned
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-muted-foreground">
              <p><span className="text-foreground">unsloth/Qwen2.5-7B-Instruct-bnb-4bit</span> — the one node actually fine-tuned in this project.</p>
              <p>
                Kept at 7B rather than 14B deliberately: this task has a
                bounded, exactly-specifiable output schema, so a synthetic
                training set could be generated programmatically — scale
                matters less than fine-tuning here, and the 7B fine-tune
                already reached 63% exact-field-match from a 30% base.
              </p>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">LoRA configuration</CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="grid grid-cols-2 gap-x-6 gap-y-2 font-mono text-xs text-muted-foreground sm:grid-cols-3">
              <div><dt className="text-foreground/70">rank</dt><dd>16</dd></div>
              <div><dt className="text-foreground/70">alpha</dt><dd>16</dd></div>
              <div><dt className="text-foreground/70">dropout</dt><dd>0</dd></div>
              <div><dt className="text-foreground/70">learning rate</dt><dd>2e-4</dd></div>
              <div><dt className="text-foreground/70">epochs</dt><dd>3</dd></div>
              <div><dt className="text-foreground/70">effective batch</dt><dd>8 (2 × 4 accum)</dd></div>
              <div><dt className="text-foreground/70">optimizer</dt><dd>adamw_8bit</dd></div>
              <div className="col-span-2 sm:col-span-3">
                <dt className="text-foreground/70">target modules</dt>
                <dd>q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj</dd>
              </div>
            </dl>
            <p className="mt-3 text-xs text-muted-foreground">
              q/k/v/o_proj are the literal query, key, value, and output
              projection matrices of multi-head self-attention — this is
              where attention shows up concretely in what was built, not
              just in a diagram.
            </p>
          </CardContent>
        </Card>
      </section>

      <Separator />

      {/* ---- Before/after ---- */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Fine-tuning: the honest before/after</h2>
        <p className="text-sm text-muted-foreground">
          Training loss converged cleanly (1.807 → 0.058 over 3 epochs, 507
          steps). All 30 validation examples, both models&rsquo; raw output,
          side by side, are in <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">finetuning/outputs/before_after_comparison.json</code>.
        </p>
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-card text-left text-muted-foreground">
                <th className="px-4 py-2 font-medium">Metric</th>
                <th className="px-4 py-2 font-medium">Base (7B, no fine-tune)</th>
                <th className="px-4 py-2 font-medium">Fine-tuned</th>
              </tr>
            </thead>
            <tbody>
              {BEFORE_AFTER.map((row) => (
                <tr key={row.metric} className="border-b border-border last:border-0">
                  <td className="px-4 py-2 font-mono text-xs">{row.metric}</td>
                  <td className="px-4 py-2 font-mono">{row.before}</td>
                  <td className="flex items-center gap-1.5 px-4 py-2 font-mono">
                    {row.after}
                    {row.afterBetter ? (
                      <CheckCircle2 className="size-3.5 text-acid" />
                    ) : (
                      <XCircle className="size-3.5 text-caveat" />
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Alert variant="caveat">
          <AlertTriangle />
          <AlertTitle>The valid_json_rate drop looks bad in isolation</AlertTitle>
          <AlertDescription>
            Zero fine-tuned examples were &ldquo;valid JSON but wrong
            schema&rdquo; — versus more than half of the base model&rsquo;s
            valid-JSON outputs being schema-wrong. The fine-tune learned the
            target structure almost perfectly whenever it produced parseable
            JSON at all. The gap is one narrow bug: malformed list-bracket
            syntax on the <code className="font-mono">comorbidities</code>{" "}
            field in ~20% of examples. A retry brings the theoretical
            failure rate toward ~1%, but live testing hit a query that
            failed all 3 retries back to back — the retry is a mitigation,
            not a fix. The agent&rsquo;s graceful-degradation path (a
            clarification response, not a crash) is what actually
            guarantees the pipeline never breaks on this.
          </AlertDescription>
        </Alert>
      </section>

      <Separator />

      {/* ---- RoPE / attention, honestly framed ---- */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">RoPE, attention, and what fine-tuning does (and doesn&rsquo;t) touch</h2>
        <Card className="border-caveat/30">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <AlertTriangle className="size-4 text-caveat" />
              Inherited, not implemented
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>
              Qwen2.5 uses rotary position embeddings (RoPE) and grouped-query
              attention internally, from pretraining. Fine-tuning does not let
              you apply or configure RoPE — you get whatever the base model
              was pretrained with.
            </p>
            <p>
              What this project actually demonstrates: choosing a model that
              uses RoPE and explaining what it does — relative position
              encoding baked directly into the attention dot-product, by
              rotating query/key vectors by an angle proportional to their
              sequence position, giving better length generalization than
              learned absolute position embeddings. Not &ldquo;implementing&rdquo;
              RoPE. The LoRA adapters trained above target the attention
              projection matrices themselves (q/k/v/o_proj) — that&rsquo;s the
              real, concrete point of contact with attention this project has.
            </p>
          </CardContent>
        </Card>

        <Card className="border-caveat/30">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <AlertTriangle className="size-4 text-caveat" />
              CLS / MLM: deliberately absent from the generative nodes
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>
              <code className="font-mono">parse_query</code> and{" "}
              <code className="font-mono">generate_report</code> are
              decoder-only generation tasks — a CLS token and a
              masked-language-modeling objective are BERT-family
              (encoder-only) concepts that don&rsquo;t fit either node, and no
              code here adds one to them.
            </p>
            <p>
              The one place an encoder/CLS-pooled model legitimately belongs
              is the Qdrant RAG retrieval step, which uses FastEmbed&rsquo;s{" "}
              <code className="font-mono">BAAI/bge-small-en</code> — BERT-derived,
              CLS-pooled, MLM-pretrained, then further tuned for embedding
              quality. That&rsquo;s the correct, motivated use of an
              encoder-style model in this project: retrieval, not generation.
            </p>
          </CardContent>
        </Card>
      </section>

      <Separator />

      {/* ---- Two ML tasks, not three ---- */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Two trained ML tasks, not three</h2>
        <p className="text-sm text-muted-foreground">
          The original plan called for three model tasks: efficacy, side-effect
          probability, and a standalone &ldquo;combination interaction&rdquo;
          classifier. With only 5 drug classes in this registry, there are
          exactly C(5,2) = 10 possible class-pair combinations — of which
          exactly 1 (ACE inhibitor + ARB) is flagged discouraged. Training a
          classifier on 10 rows with 1 positive would be memorization dressed
          up as ML.
        </p>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">What actually happened</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              The combination-interaction signal was folded into task 1
              (efficacy): its training set includes both single-drug and
              2-drug combination rows, so the model has to learn the
              difference between a single and a combined response directly
              from data — that <em>is</em> the interaction task, just not a
              separate head.
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">The safety flag is a rule, not a model</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              <code className="font-mono">DISCOURAGED_COMBINATIONS</code> stays
              hard-coded in the drug registry, not a model output. Whether
              dual RAAS blockade is dangerous is a fixed clinical fact, not a
              statistical pattern with useful uncertainty — a model
              confidence score would understate how well-established the
              risk actually is.
            </CardContent>
          </Card>
        </div>
        <p className="text-sm text-muted-foreground">
          Net result: two trained ML tasks (efficacy regression, side-effect
          multi-label classification) plus one rule-based safety check —
          rather than three ML tasks where one would have been theater.
        </p>
      </section>

      <Separator />

      {/* ---- GGUF abandonment ---- */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">The GGUF export that had to be abandoned</h2>
        <p className="text-sm text-muted-foreground">
          The original serving plan for <code className="font-mono">parse_query</code> matched{" "}
          <code className="font-mono">generate_report</code>&rsquo;s pattern: merge the LoRA
          adapter into the base model, export to GGUF, serve via Ollama. The
          merge step succeeded — a 15GB fp16 safetensors model, verified
          intact on disk. Converting that model to GGUF crashed the host
          machine&rsquo;s IDE three separate times via kernel out-of-memory,
          despite a memory cap and a self-kill watchdog — the export&rsquo;s
          write phase spiked memory faster than either safety net could
          reliably catch.
        </p>
        <div className="flex flex-col gap-3 sm:flex-row">
          <Card className="flex-1 border-magenta/30">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <XCircle className="size-4 text-magenta" />
                Abandoned: GGUF export → Ollama
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              Write-heavy: dirty pages from writing a new 8–15GB file need
              flushing before that memory is reclaimable. This is the actual
              mechanism behind the crashes, not something a tighter memory
              cap alone could fix.
            </CardContent>
          </Card>
          <Card className="flex-1 border-acid/30">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <CheckCircle2 className="size-4 text-acid" />
                Used instead: direct in-process 4-bit inference
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              Read-heavy: memory-mapped safetensors, reclaimable page cache.
              Verified stable (23–25GB available throughout) in two isolated
              test runs before being wired into the real agent node.
            </CardContent>
          </Card>
        </div>
        <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <ArrowRight className="size-3.5" />
          Lesson: when a write-heavy operation is crashing a shared
          environment, changing degree (smaller quant, tighter cap, faster
          polling) is not the same as changing kind. The fix was recognizing
          these are different memory access patterns, not tuning the write
          path harder.
        </p>
        <p className="text-sm text-muted-foreground">
          Disclosed tradeoff: the fine-tuned model needs ~5GB in 4-bit;{" "}
          <code className="font-mono">generate_report</code>&rsquo;s Ollama-served
          14B model needs ~9GB — together over the 12GB card. Since the two
          nodes never run concurrently, the fine-tuned model loads,
          generates, and explicitly frees VRAM on every single call rather
          than staying resident, trading load latency (part of the ~35–40s
          above) for the two models never fighting over VRAM.
        </p>
      </section>
    </div>
  );
}
