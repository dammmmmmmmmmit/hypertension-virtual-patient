import { AlertTriangle, ShieldAlert } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { HYPERTENSION_DRUGS } from "@/lib/drug-registry";
import type { DrugClass } from "@/lib/types";

export const metadata = {
  title: "Domain — Virtual Patient Drug-Response Simulator",
};

// Class-level mean BP reduction at standard dose (DECISIONS.md #1) —
// re-derived from primary sources, not from memory: Law MR, Morris JK,
// Wald NJ. "Use of blood pressure lowering drugs in the prevention of
// cardiovascular disease: meta-analysis of 147 randomised trials..."
// BMJ 2009;338:b1665. This is the training label ANCHOR — per-compound
// predictions are a small, disclosed heuristic adjustment on top of
// these class averages, not independently measured per drug.
const DRUG_CLASSES: {
  key: DrugClass;
  label: string;
  gene: string;
  mechanism: string;
  systolic: number;
  diastolic: number;
}[] = [
  {
    key: "ace_inhibitor",
    label: "ACE inhibitor",
    gene: "ACE",
    mechanism:
      "Blocks angiotensin-converting enzyme, reducing formation of angiotensin II and aldosterone release.",
    systolic: 8.5,
    diastolic: 4.7,
  },
  {
    key: "arb",
    label: "ARB",
    gene: "AGTR1",
    mechanism:
      "Blocks the type-1 angiotensin II receptor directly, preventing angiotensin II from causing vasoconstriction and aldosterone release.",
    systolic: 10.3,
    diastolic: 5.7,
  },
  {
    key: "beta_blocker",
    label: "Beta-blocker",
    gene: "ADRB1",
    mechanism:
      "Blocks β1-adrenergic receptors, reducing heart rate, cardiac contractility, and renin release.",
    systolic: 9.2,
    diastolic: 6.7,
  },
  {
    key: "calcium_channel_blocker",
    label: "Calcium-channel blocker",
    gene: "CACNA1C",
    mechanism:
      "Blocks voltage-gated L-type calcium channels in vascular smooth muscle, causing vasodilation.",
    systolic: 8.8,
    diastolic: 5.9,
  },
  {
    key: "thiazide_diuretic",
    label: "Thiazide diuretic",
    gene: "SLC12A3",
    mechanism:
      "Inhibits the Na⁺/Cl⁻ cotransporter in the distal convoluted tubule, reducing sodium reabsorption and blood volume.",
    systolic: 8.8,
    diastolic: 4.4,
  },
];

const COMORBIDITIES = [
  {
    label: "Type 2 diabetes",
    note: "Tracked as a patient covariate; no dedicated adjustment channel beyond the standard renal-function pathway.",
  },
  {
    label: "Chronic kidney disease",
    note: "Feeds the eGFR-based renal adjustment factor — a disclosed multiplier on the efficacy prediction, not validated PK/PD modeling.",
  },
  {
    label: "Heart failure",
    note: "Tracked as a patient covariate for report context; not a separate model input channel.",
  },
  {
    label: "Asthma / COPD",
    note: "Relevant mainly as a beta-blocker caution — a side-effect-risk signal, not a BP-magnitude effect, so it shows up in the side-effect head rather than the efficacy prediction.",
  },
];

export default function DomainPage() {
  return (
    <div className="mx-auto max-w-5xl space-y-12 px-6 py-10">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">Domain</h1>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Modeling &ldquo;a virtual patient&rdquo; in general terms is out of
          reach for a project this size. This tool models{" "}
          <span className="text-foreground">hypertension only</span> — five
          drug classes, blood-pressure/heart-rate/renal-function parameters,
          and patient covariates treated as adjustment factors rather than a
          full mechanistic physiology model.
        </p>
      </header>

      <Alert variant="caveat">
        <AlertTriangle />
        <AlertTitle>Screening aid, not point-of-care guidance</AlertTitle>
        <AlertDescription>
          This is a research-screening tool for exploring plausible
          drug-response directions, not a clinical decision-support system.
          Every generated report says so explicitly, and every prediction
          here is downstream of class-level trial evidence plus a small,
          disclosed heuristic — never presented with more confidence than
          that warrants.
        </AlertDescription>
      </Alert>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Five drug classes, fifteen drugs</h2>
        <p className="text-sm text-muted-foreground">
          Three drugs per class. Systolic/diastolic figures are mean
          reduction at standard dose from Law, Morris &amp; Wald&rsquo;s
          147-trial meta-analysis (BMJ 2009) — the training label anchor
          every per-compound prediction is adjusted from, not measured
          independently.
        </p>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {DRUG_CLASSES.map((cls) => (
            <Card key={cls.key}>
              <CardHeader>
                <CardTitle className="flex items-center justify-between text-base">
                  {cls.label}
                  <Badge variant="outline" className="font-mono">
                    {cls.gene}
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground">{cls.mechanism}</p>
                <div className="flex gap-4 font-mono text-xs">
                  <span className="text-acid">
                    −{cls.systolic} <span className="text-muted-foreground">mmHg SBP</span>
                  </span>
                  <span className="text-acid">
                    −{cls.diastolic} <span className="text-muted-foreground">mmHg DBP</span>
                  </span>
                </div>
                <Separator />
                <div className="flex flex-wrap gap-1.5">
                  {HYPERTENSION_DRUGS.filter((d) => d.drugClass === cls.key).map((d) => (
                    <Badge key={d.name} variant="secondary" className="capitalize">
                      {d.name}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      <Separator />

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">The one hard-coded safety rule</h2>
        <Alert variant="discouraged">
          <ShieldAlert />
          <AlertTitle>ACE inhibitor + ARB (dual RAAS blockade)</AlertTitle>
          <AlertDescription className="text-white/90">
            Combining an ACE inhibitor and an ARB is flagged as discouraged —
            elevated hyperkalemia and renal-impairment risk. This is a
            hard-coded clinical rule read from the drug registry, not a
            model prediction: whether dual RAAS blockade is dangerous is a
            fixed clinical fact, not a statistical pattern with useful
            uncertainty. Presenting it with a model confidence score would
            understate how well-established the risk actually is.
          </AlertDescription>
        </Alert>
      </section>

      <Separator />

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Patient covariates</h2>
        <p className="text-sm text-muted-foreground">
          Comorbidities and vitals are tracked as adjustment factors on top
          of the class-level prediction, not inputs to a mechanistic
          physiology model.
        </p>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {COMORBIDITIES.map((c) => (
            <Card key={c.label}>
              <CardContent className="pt-6">
                <p className="text-sm font-medium text-foreground">{c.label}</p>
                <p className="mt-1 text-sm text-muted-foreground">{c.note}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}
