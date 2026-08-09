"use client";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { COMORBIDITY_LABELS } from "@/lib/drug-registry";
import type { Comorbidity, PatientProfile } from "@/lib/types";

const COMORBIDITY_OPTIONS: Comorbidity[] = [
  "type2_diabetes",
  "chronic_kidney_disease",
  "heart_failure",
  "asthma_copd",
];

interface PatientFormProps {
  value: PatientProfile;
  onChange: (value: PatientProfile) => void;
}

/**
 * Structured patient-vitals form — deliberately NOT parsed from free
 * text. Mirrors app/ui/streamlit_app.py's sidebar fields exactly (same
 * defaults, same ranges from app/schemas/patient.py's Field(...)
 * bounds). See ui_build_brief.md: numeric fields that feed the
 * prediction model shouldn't depend on the fine-tuned model's real 63%
 * exact-field-match rate (DECISIONS.md #7) — this split is deliberate,
 * not a UI shortcut.
 */
export function PatientForm({ value, onChange }: PatientFormProps) {
  function update<K extends keyof PatientProfile>(key: K, val: PatientProfile[K]) {
    onChange({ ...value, [key]: val });
  }

  function updateBaseline<K extends keyof PatientProfile["baseline"]>(
    key: K,
    val: PatientProfile["baseline"][K]
  ) {
    onChange({ ...value, baseline: { ...value.baseline, [key]: val } });
  }

  function toggleComorbidity(c: Comorbidity, checked: boolean) {
    const withoutNone = value.comorbidities.filter((x) => x !== "none");
    const next = checked
      ? [...withoutNone, c]
      : withoutNone.filter((x) => x !== c);
    update("comorbidities", next.length ? next : ["none"]);
  }

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1.5">
          <Label htmlFor="age">Age</Label>
          <Input
            id="age"
            type="number"
            min={18}
            max={100}
            value={value.age}
            onChange={(e) => update("age", Number(e.target.value))}
            className="font-mono"
          />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="sex">Sex</Label>
          <Select
            value={value.sex}
            onValueChange={(v) => update("sex", v as PatientProfile["sex"])}
          >
            <SelectTrigger id="sex" className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="male">Male</SelectItem>
              <SelectItem value="female">Female</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="space-y-1.5">
        <Label htmlFor="weight">Weight (kg)</Label>
        <Input
          id="weight"
          type="number"
          min={30}
          max={250}
          step={0.1}
          value={value.weight_kg}
          onChange={(e) => update("weight_kg", Number(e.target.value))}
          className="font-mono"
        />
      </div>

      <div>
        <p className="mb-2 text-xs font-medium tracking-wide text-indigo uppercase">
          Baseline vitals
        </p>
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <Label htmlFor="sbp">Systolic BP (mmHg)</Label>
            <Input
              id="sbp"
              type="number"
              min={70}
              max={250}
              value={value.baseline.systolic_bp}
              onChange={(e) => updateBaseline("systolic_bp", Number(e.target.value))}
              className="font-mono"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="dbp">Diastolic BP (mmHg)</Label>
            <Input
              id="dbp"
              type="number"
              min={40}
              max={150}
              value={value.baseline.diastolic_bp}
              onChange={(e) => updateBaseline("diastolic_bp", Number(e.target.value))}
              className="font-mono"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="hr">Heart rate (bpm)</Label>
            <Input
              id="hr"
              type="number"
              min={30}
              max={200}
              value={value.baseline.heart_rate}
              onChange={(e) => updateBaseline("heart_rate", Number(e.target.value))}
              className="font-mono"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="egfr">eGFR (renal function)</Label>
            <Input
              id="egfr"
              type="number"
              min={5}
              max={140}
              value={value.baseline.egfr}
              onChange={(e) => updateBaseline("egfr", Number(e.target.value))}
              className="font-mono"
            />
          </div>
        </div>
        <div className="mt-3 space-y-1.5">
          <Label htmlFor="potassium">Serum potassium (mmol/L)</Label>
          <Input
            id="potassium"
            type="number"
            min={2.5}
            max={7.0}
            step={0.1}
            value={value.baseline.serum_potassium}
            onChange={(e) => updateBaseline("serum_potassium", Number(e.target.value))}
            className="font-mono max-w-[10rem]"
          />
        </div>
      </div>

      <div>
        <p className="mb-2 text-xs font-medium tracking-wide text-indigo uppercase">
          Comorbidities
        </p>
        <div className="flex flex-wrap gap-1.5">
          {COMORBIDITY_OPTIONS.map((c) => {
            const active = value.comorbidities.includes(c);
            return (
              <button
                key={c}
                type="button"
                aria-pressed={active}
                onClick={() => toggleComorbidity(c, !active)}
                className={`rounded-full border px-3 py-1 text-xs transition-colors ${
                  active
                    ? "border-indigo/50 bg-indigo/15 text-foreground"
                    : "border-border bg-transparent text-muted-foreground hover:border-indigo/30 hover:text-foreground"
                }`}
              >
                {COMORBIDITY_LABELS[c]}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export const DEFAULT_PATIENT: PatientProfile = {
  age: 58,
  sex: "male",
  weight_kg: 82,
  baseline: {
    systolic_bp: 152,
    diastolic_bp: 96,
    heart_rate: 78,
    serum_potassium: 4.2,
    egfr: 90,
  },
  comorbidities: ["none"],
  current_medications: [],
};
