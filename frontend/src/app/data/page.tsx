import { AlertTriangle } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

export const metadata = {
  title: "Data & methodology — Virtual Patient Drug-Response Simulator",
};

export default function DataPage() {
  return (
    <div className="mx-auto max-w-5xl space-y-12 px-6 py-10">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">Data &amp; methodology</h1>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Real records pulled from this project&rsquo;s own cached ChEMBL,
          SIDER, and training data — not illustrative examples. Includes the
          two genuine data gaps and the honest evaluation numbers, not just
          the parts that turned out clean.
        </p>
      </header>

      {/* ---- Real ChEMBL record ---- */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">A real ChEMBL bioactivity record</h2>
        <p className="text-sm text-muted-foreground">
          Losartan, as cached in this project&rsquo;s Postgres{" "}
          <code className="font-mono">resolved_compounds</code> table —{" "}
          <code className="font-mono">CHEMBL191</code>, resolved via an exact{" "}
          <code className="font-mono">pref_name__iexact</code> match after a
          fuzzy-search bug initially resolved it to an unrelated compound,{" "}
          <code className="font-mono">CHEMBL382821</code>
          &nbsp;(&ldquo;losartan nitrooxy ester&rdquo;) — see the architecture
          page&rsquo;s pivot story for the parallel GGUF incident; this was an
          earlier live-testing catch of the same kind.
        </p>
        <Card>
          <CardContent className="overflow-x-auto pt-6">
            <table className="w-full text-sm">
              <tbody>
                {[
                  ["name", "losartan"],
                  ["drug_class", "arb"],
                  ["gene_symbol", "AGTR1"],
                  ["chembl_id", "CHEMBL191"],
                  ["target_chembl_id", "CHEMBL227"],
                  [
                    "canonical_smiles",
                    "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1",
                  ],
                  ["mean_potency (pX)", "7.98"],
                  ["n_valid_potency_records", "30"],
                  ["pubchem_cid", "3961"],
                ].map(([k, v]) => (
                  <tr key={k} className="border-b border-border last:border-0">
                    <td className="w-56 py-1.5 pr-4 font-mono text-xs text-muted-foreground">
                      {k}
                    </td>
                    <td className="py-1.5 font-mono text-xs break-all text-foreground">
                      {v}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      </section>

      <Separator />

      {/* ---- Real SIDER record ---- */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">A real SIDER record</h2>
        <p className="text-sm text-muted-foreground">
          SIDER identifies losartan by its flat STITCH ID{" "}
          <code className="font-mono">CID100003961</code> (from SIDER&rsquo;s
          own <code className="font-mono">drug_names.tsv</code> — joined by
          name, not by independently re-deriving a PubChem CID offset, after
          that approach broke on lisinopril; see the design decisions log).
          Raw rows from <code className="font-mono">meddra_all_se.tsv.gz</code>:
        </p>
        <Card>
          <CardContent className="overflow-x-auto pt-6">
            <pre className="font-mono text-xs whitespace-pre text-muted-foreground">
{`CID100003961  CID000003961  C0000737  LLT  C0000737  Abdominal pain
CID100003961  CID000003961  C0000737  PT   C0687713  Gastrointestinal pain
CID100003961  CID000003961  C0001824  LLT  C0001824  Agranulocytosis
CID100003961  CID000003961  C0001883  PT   C0600260  Obstructive airways disorder`}
            </pre>
          </CardContent>
        </Card>
        <p className="text-sm text-muted-foreground">
          SIDER&rsquo;s underlying data comes from mining FDA drug label
          adverse-event sections — a snapshot from years ago, not a
          live-updated feed. Combined with a very low bar for label
          inclusion, this is exactly what produced the near-universal-term
          problem below.
        </p>
      </section>

      <Separator />

      {/* ---- Two real data gaps ---- */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Two genuine data gaps</h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <Card className="border-caveat/30">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <AlertTriangle className="size-4 text-caveat" />
                Thiazide potency: 0/3 drugs
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-muted-foreground">
              <p>
                Hydrochlorothiazide, chlorthalidone, and indapamide all have{" "}
                <code className="font-mono">n_valid_potency_records = 0</code>{" "}
                in ChEMBL — thiazides act on a transporter, not the kind of
                receptor-binding assay ChEMBL is built around, so this is a
                real coverage gap in the source data, not a fetch bug.
              </p>
              <p>
                Handled by setting the potency z-score to 0 for these three
                (&ldquo;assume class-average&rdquo;), not by imputing a
                fabricated number.
              </p>
            </CardContent>
          </Card>
          <Card className="border-caveat/30">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <AlertTriangle className="size-4 text-caveat" />
                Enalapril: 0 SIDER side effects
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-muted-foreground">
              <p>
                14/15 registry drugs have SIDER coverage; enalapril returns
                an empty side-effect list. Checked directly rather than
                assumed — this is a genuine SIDER coverage gap, not an
                artifact of the STITCH-ID join method (verified against the
                same name-based join that correctly resolves the other 14).
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      <Separator />

      {/* ---- Training data ---- */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Training data, real rows</h2>
        <p className="text-sm text-muted-foreground">
          60 efficacy rows (15 single-drug + 45 combinations), 14 side-effect
          rows against a 29-term filtered vocabulary. One real row from each,
          exactly as trained on:
        </p>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium text-muted-foreground">
              efficacy_dataset.csv — lisinopril
            </CardTitle>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <pre className="font-mono text-xs whitespace-pre text-muted-foreground">
{`label       n_drugs  discouraged  systolic_delta  diastolic_delta  mean_potency_z
lisinopril  1        False        8.95            4.95             0.664`}
            </pre>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium text-muted-foreground">
              side_effect_dataset.csv — lisinopril (of 29 label columns, first few)
            </CardTitle>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <pre className="font-mono text-xs whitespace-pre text-muted-foreground">
{`label       se__Agranulocytosis  se__Anxiety  se__Arthralgia  se__Cough  se__Dysgeusia
lisinopril  1                    1            1               1          1`}
            </pre>
          </CardContent>
        </Card>
        <p className="text-xs text-muted-foreground">
          The vocabulary was filtered down from an initial top-N-by-frequency
          pass, which was dominated by terms present in 14/14 drugs
          regardless of mechanism (see design decisions log #4) — the current
          29 terms all have real variance across the registry.
        </p>
      </section>

      <Separator />

      {/* ---- Label methodology ---- */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">How the efficacy label was built</h2>
        <p className="text-sm text-muted-foreground">
          There is no public per-compound RCT dataset covering all 15
          registry drugs at consistent dosing, so the efficacy label is{" "}
          <span className="text-foreground">semi-synthetic</span>: a
          real class-level trial anchor, adjusted by a small, disclosed
          heuristic for each compound&rsquo;s relative potency within its
          class.
        </p>
        <Card>
          <CardContent className="pt-6">
            <pre className="overflow-x-auto font-mono text-xs text-muted-foreground">
{`baseline_delta(class)   = class-level trial mean (Law/Morris/Wald, BMJ 2009)
potency_z(drug)         = (drug's pX - class mean pX) / class std pX
adjusted_delta(drug)    = baseline_delta(class) * (1 + ALPHA * clip(potency_z, -2, 2))
                          ALPHA = 0.08 — hand-set, not fit from data`}
            </pre>
          </CardContent>
        </Card>
        <p className="text-sm text-muted-foreground">
          <span className="text-foreground">ALPHA = 0.08 is deliberately
          small and documented, not learned</span> — there is no real
          per-compound ground truth to fit it against. This is disclosed here
          because it&rsquo;s the single biggest scientific-honesty tradeoff
          in the project: the model&rsquo;s output should never be presented
          with more confidence than &ldquo;class-level trial evidence, plus a
          small heuristic adjustment for this compound&rsquo;s relative
          potency.&rdquo;
        </p>
      </section>

      <Separator />

      {/* ---- Honest evaluation numbers ---- */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Evaluation — reported honestly</h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between text-base">
                Efficacy model
                <Badge variant="outline" className="font-mono">
                  n=60, LOO-CV
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <p className="font-mono text-sm text-acid">
                MAE 0.171 mmHg (SBP) / 0.295 mmHg (DBP)
              </p>
              <p className="text-xs text-muted-foreground">
                Labels are semi-synthetic — this low error reflects the model
                recovering a smooth, low-noise constructed function, not
                validated real-world predictive accuracy.
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between text-base">
                Side-effect model
                <Badge variant="outline" className="font-mono">
                  n=14, LOO-CV
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <p className="font-mono text-sm text-acid">
                Mean per-label accuracy 0.729
              </p>
              <p className="text-xs text-muted-foreground">
                14 samples across 30 labels is a genuinely small-data
                problem — this figure is inflated by near-constant labels
                (e.g. a side effect present in 13/14 drugs is &ldquo;accurate&rdquo;
                even predicting the majority class). Treat this as a
                smoothed lookup over structural similarity, not a validated
                QSAR model.
              </p>
            </CardContent>
          </Card>
        </div>
        <Alert variant="caveat">
          <AlertTriangle />
          <AlertTitle>Neither number should be read as clinical accuracy</AlertTitle>
          <AlertDescription>
            Both models are evaluated against constructed/small-sample
            labels, not real per-patient clinical outcomes — a low LOO-CV
            error is a statement about how well the model recovers the
            training construction, not about real-world predictive validity.
          </AlertDescription>
        </Alert>
      </section>
    </div>
  );
}
