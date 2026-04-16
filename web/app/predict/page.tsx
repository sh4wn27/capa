"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  FlaskConical,
  ChevronDown,
  Loader2,
  AlertCircle,
  CheckCircle2,
  XCircle,
  RotateCcw,
  Download,
  Info,
  Sparkles,
  WifiOff,
} from "lucide-react";
import { Button }      from "@/components/ui/button";
import { Input }       from "@/components/ui/input";
import { Label }       from "@/components/ui/label";
import { Badge }       from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { HLACombobox } from "@/components/HLACombobox";
import RiskChart        from "@/components/RiskChart";
import AttentionHeatmap from "@/components/AttentionHeatmap";
import { predictRisk, type PredictionRequest, type PredictionResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

/* ─── constants ─────────────────────────────────────────────────────── */
const LOCI = ["A", "B", "C", "DRB1", "DQB1"] as const;
type Locus = (typeof LOCI)[number];

const ease = [0.21, 0.47, 0.32, 0.98] as [number, number, number, number];

/* ─── example case (10/10 mismatched unrelated, paediatric ALL) ──────── */
const EXAMPLE_HLA = {
  donor: {
    A:    "A*02:01",
    B:    "B*07:02",
    C:    "C*07:02",
    DRB1: "DRB1*15:01",
    DQB1: "DQB1*06:02",
  } as Record<Locus, string>,
  recipient: {
    A:    "A*03:01",
    B:    "B*08:01",
    C:    "C*07:01",
    DRB1: "DRB1*03:01",
    DQB1: "DQB1*02:01",
  } as Record<Locus, string>,
};

const EXAMPLE_CLINICAL = {
  ageRecipient:   "8",
  ageDonor:       "35",
  cd34Dose:       "5.2",
  disease:        "ALL",
  conditioning:   "myeloablative",
  donorType:      "matched_unrelated",
  stemCellSource: "peripheral_blood",
};

/* Weibull CIF helper */
function weibullCIF(t: number, shape: number, scale: number) {
  return 1 - Math.exp(-Math.pow(t / scale, shape));
}

/* Pre-computed mock response for "Try Example" instant demo */
function buildMockResult(): PredictionResponse {
  const N = 100;
  const days = Array.from({ length: N }, (_, i) => (i / (N - 1)) * 730);
  return {
    gvhd: {
      cumulative_incidence: days.map((d) => weibullCIF(d, 1.4, 280)),
      risk_score: 0.48,
      time_points: days,
    },
    relapse: {
      cumulative_incidence: days.map((d) => weibullCIF(d, 1.1, 490)),
      risk_score: 0.29,
      time_points: days,
    },
    trm: {
      cumulative_incidence: days.map((d) => weibullCIF(d, 0.85, 650)),
      risk_score: 0.17,
      time_points: days,
    },
    attention_weights: [
      [0.44, 0.12, 0.09, 0.27, 0.08],  // Donor A
      [0.14, 0.39, 0.10, 0.21, 0.16],  // Donor B
      [0.10, 0.12, 0.51, 0.13, 0.14],  // Donor C
      [0.30, 0.17, 0.11, 0.28, 0.14],  // Donor DRB1
      [0.08, 0.14, 0.12, 0.21, 0.45],  // Donor DQB1
    ],
    mismatch_count: 5,
    model_version:  "capa-v0.1-demo",
  };
}

/* ─── risk level classification ─────────────────────────────────────── */
type RiskLevel = "low" | "medium" | "high";

function classifyRisk(score: number): RiskLevel {
  if (score < 0.25) return "low";
  if (score < 0.50) return "medium";
  return "high";
}

const LEVEL_CONFIG: Record<RiskLevel, { label: string; bar: string; bg: string; border: string; text: string; icon: typeof CheckCircle2 }> = {
  low: {
    label:  "Low",
    bar:    "bg-green-500",
    bg:     "bg-green-50",
    border: "border-green-200",
    text:   "text-green-700",
    icon:   CheckCircle2,
  },
  medium: {
    label:  "Medium",
    bar:    "bg-amber-400",
    bg:     "bg-amber-50",
    border: "border-amber-200",
    text:   "text-amber-700",
    icon:   AlertCircle,
  },
  high: {
    label:  "High",
    bar:    "bg-blush-500",
    bg:     "bg-red-50",
    border: "border-red-200",
    text:   "text-red-700",
    icon:   XCircle,
  },
};

/* Event display config */
const EVENT_CONFIG = {
  gvhd: {
    label:    "GvHD",
    full:     "Graft-versus-Host Disease",
    color:    "#E69F00",
    colorCls: "text-amber-600",
  },
  relapse: {
    label:    "Relapse",
    full:     "Disease Relapse",
    color:    "#0072B2",
    colorCls: "text-blue-600",
  },
  trm: {
    label:    "TRM",
    full:     "Transplant-Related Mortality",
    color:    "#D55E00",
    colorCls: "text-orange-700",
  },
} as const;

/* ─── sub-components ────────────────────────────────────────────────── */

function RiskGauge({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const level = classifyRisk(score);
  const cfg = LEVEL_CONFIG[level];
  const Icon = cfg.icon;

  return (
    <div className={cn("rounded-xl border p-5", cfg.bg, cfg.border)}>
      <div className="flex items-center justify-between mb-3">
        <Icon className={cn("h-4 w-4", cfg.text)} />
        <Badge
          className={cn("text-[10px] uppercase tracking-wider border", cfg.bg, cfg.border, cfg.text)}
        >
          {cfg.label} risk
        </Badge>
      </div>

      {/* Score */}
      <p className={cn("text-3xl font-black tabular-nums", cfg.text)}>
        {pct}
        <span className="text-lg font-semibold">%</span>
      </p>
      <p className="text-xs text-muted-foreground mt-0.5">cumulative at day 730</p>

      {/* Progress bar */}
      <div className="mt-3 h-1.5 rounded-full bg-black/8 overflow-hidden">
        <motion.div
          className={cn("h-full rounded-full", cfg.bar)}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.7, ease, delay: 0.1 }}
        />
      </div>

      {/* Timepoint snapshots */}
      <div className="mt-3 grid grid-cols-3 gap-1 text-center">
        {(["d100", "d365", "d730"] as const).map((k) => (
          <div key={k}>
            <p className={cn("text-xs font-semibold", cfg.text)}>—</p>
            <p className="text-[10px] text-muted-foreground">{k}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function RiskSummaryCards({ result }: { result: PredictionResponse }) {
  const events = [
    { key: "gvhd",    data: result.gvhd    },
    { key: "relapse", data: result.relapse  },
    { key: "trm",     data: result.trm      },
  ] as const;

  const N = result.gvhd.cumulative_incidence.length;
  function snap(arr: number[], dayFraction: number) {
    const idx = Math.min(Math.round(dayFraction * (N - 1)), N - 1);
    return (arr[idx] * 100).toFixed(1);
  }

  return (
    <div className="grid gap-4 sm:grid-cols-3">
      {events.map(({ key, data }, ei) => {
        const cfg = EVENT_CONFIG[key];
        const level = classifyRisk(data.risk_score);
        const lcfg = LEVEL_CONFIG[level];
        const Icon = lcfg.icon;
        const cif = data.cumulative_incidence;

        return (
          <motion.div
            key={key}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease, delay: ei * 0.1 }}
          >
            <Card className={cn("border", lcfg.border, "overflow-hidden")}>
              {/* Colored top bar */}
              <div className="h-1" style={{ backgroundColor: cfg.color }} />
              <CardContent className="pt-5 pb-5">
                {/* Header */}
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <p className="font-semibold text-navy text-sm">{cfg.full}</p>
                    <p className="text-xs text-muted-foreground">{cfg.label}</p>
                  </div>
                  <div className={cn("flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold", lcfg.bg, lcfg.text)}>
                    <Icon className="h-3 w-3" />
                    {lcfg.label}
                  </div>
                </div>

                {/* Big score */}
                <p className={cn("text-4xl font-black tabular-nums", cfg.colorCls)}>
                  {Math.round(data.risk_score * 100)}
                  <span className="text-xl font-semibold">%</span>
                </p>
                <p className="text-xs text-muted-foreground mb-3">risk score (day 730)</p>

                {/* Progress bar */}
                <div className="h-1.5 rounded-full bg-black/8 overflow-hidden mb-4">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ backgroundColor: cfg.color }}
                    initial={{ width: 0 }}
                    animate={{ width: `${data.risk_score * 100}%` }}
                    transition={{ duration: 0.75, ease, delay: 0.15 + ei * 0.1 }}
                  />
                </div>

                {/* Day snapshots */}
                <div className="grid grid-cols-3 gap-2 text-center">
                  {([
                    { day: "100d",  frac: 100 / 730  },
                    { day: "365d",  frac: 365 / 730  },
                    { day: "730d",  frac: 1           },
                  ] as const).map(({ day, frac }) => (
                    <div key={day} className={cn("rounded-lg py-1.5 px-1", lcfg.bg)}>
                      <p className={cn("text-xs font-bold tabular-nums", lcfg.text)}>
                        {snap(cif, frac)}%
                      </p>
                      <p className="text-[10px] text-muted-foreground">{day}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        );
      })}
    </div>
  );
}

function MismatchSummary({
  donor,
  recipient,
}: {
  donor: Record<string, string>;
  recipient: Record<string, string>;
}) {
  const loci = LOCI.filter((l) => donor[l] || recipient[l]);
  if (loci.length === 0) return null;

  const mismatches = loci.filter((l) => donor[l] && recipient[l] && donor[l] !== recipient[l]);

  return (
    <div className="rounded-lg border border-border bg-muted/40 p-4">
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs font-semibold text-navy uppercase tracking-wider">
          Mismatch summary
        </p>
        <Badge variant={mismatches.length > 2 ? "blush" : "navy"} className="text-[10px]">
          {mismatches.length}/{loci.length} mismatched
        </Badge>
      </div>
      <div className="grid grid-cols-5 gap-2">
        {LOCI.map((l) => {
          const d = donor[l];
          const r = recipient[l];
          const matched = d && r && d === r;
          const mismatch = d && r && d !== r;
          const missing = !d || !r;
          return (
            <div
              key={l}
              className={cn(
                "rounded-lg border p-2 text-center text-[10px]",
                matched  ? "border-green-200  bg-green-50  text-green-700" :
                mismatch ? "border-red-200    bg-red-50    text-red-700"   :
                           "border-border     bg-background text-muted-foreground"
              )}
            >
              <p className="font-mono font-bold text-xs mb-0.5">{l}</p>
              {matched  && <CheckCircle2 className="h-3 w-3 mx-auto text-green-500" />}
              {mismatch && <XCircle      className="h-3 w-3 mx-auto text-red-500" />}
              {missing  && <span className="text-muted-foreground/50">—</span>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ─── HLA field group ────────────────────────────────────────────────── */
function HLAFieldGroup({
  side,
  values,
  onChange,
}: {
  side: "donor" | "recipient";
  values: Record<string, string>;
  onChange: (locus: string, val: string) => void;
}) {
  const isDonor = side === "donor";
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Badge
          variant={isDonor ? "navy" : "blush"}
          className="text-[10px] uppercase tracking-wider"
        >
          {isDonor ? "Donor" : "Recipient"}
        </Badge>
        <span className="text-xs text-muted-foreground">
          {isDonor ? "stem-cell donor" : "transplant recipient"}
        </span>
      </div>
      {LOCI.map((locus) => (
        <div key={locus} className="flex items-center gap-2">
          <Label className="w-12 text-xs text-muted-foreground font-mono shrink-0">
            {locus}
          </Label>
          <HLACombobox
            locus={locus}
            value={values[locus] ?? ""}
            onChange={(v) => onChange(locus, v)}
            className="flex-1"
          />
        </div>
      ))}
    </div>
  );
}

/* ─── page ──────────────────────────────────────────────────────────── */
export default function PredictPage() {
  /* Form state */
  const [donorHla,     setDonorHla]     = useState<Record<string, string>>({});
  const [recipientHla, setRecipientHla] = useState<Record<string, string>>({});
  const [ageRecipient, setAgeRecipient] = useState("");
  const [ageDonor,     setAgeDonor]     = useState("");
  const [cd34Dose,     setCd34Dose]     = useState("");
  const [disease,      setDisease]      = useState("");
  const [conditioning, setConditioning] = useState("");
  const [donorType,    setDonorType]    = useState("");
  const [stemCellSrc,  setStemCellSrc]  = useState("");
  const [showClinical, setShowClinical] = useState(false);

  /* Result state */
  const [result,  setResult]  = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  /* Demo-mode: true when the Python backend is unreachable */
  const [isDemoMode, setIsDemoMode] = useState(false);

  const resultsRef = useRef<HTMLDivElement>(null);

  /* Poll backend health once on mount */
  useEffect(() => {
    let cancelled = false;
    fetch("/api/predict", { method: "GET" })
      .then((r) => { if (!cancelled) setIsDemoMode(!r.ok); })
      .catch(() => { if (!cancelled) setIsDemoMode(true); });
    return () => { cancelled = true; };
  }, []);

  /* Try Example — fill + show instant mock result */
  function fillExample() {
    setDonorHla({ ...EXAMPLE_HLA.donor });
    setRecipientHla({ ...EXAMPLE_HLA.recipient });
    setAgeRecipient(EXAMPLE_CLINICAL.ageRecipient);
    setAgeDonor(EXAMPLE_CLINICAL.ageDonor);
    setCd34Dose(EXAMPLE_CLINICAL.cd34Dose);
    setDisease(EXAMPLE_CLINICAL.disease);
    setConditioning(EXAMPLE_CLINICAL.conditioning);
    setDonorType(EXAMPLE_CLINICAL.donorType);
    setStemCellSrc(EXAMPLE_CLINICAL.stemCellSource);
    setShowClinical(true);
    setError(null);
    setResult(buildMockResult());
    setTimeout(() => {
      resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 150);
  }

  function clearAll() {
    setDonorHla({});
    setRecipientHla({});
    setAgeRecipient(""); setAgeDonor(""); setCd34Dose("");
    setDisease(""); setConditioning(""); setDonorType(""); setStemCellSrc("");
    setResult(null); setError(null);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      if (isDemoMode) {
        // Backend offline — return mock result immediately
        await new Promise((r) => setTimeout(r, 600));
        setResult(buildMockResult());
        setTimeout(() => {
          resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 100);
        return;
      }
      const req: PredictionRequest = {
        donor_hla:     donorHla,
        recipient_hla: recipientHla,
        clinical: {
          age_recipient:   ageRecipient ? Number(ageRecipient) : undefined,
          age_donor:       ageDonor     ? Number(ageDonor)     : undefined,
          cd34_dose:       cd34Dose     ? Number(cd34Dose)     : undefined,
          disease:         disease       || undefined,
          conditioning:    conditioning  || undefined,
          donor_type:      donorType     || undefined,
          stem_cell_source: stemCellSrc  || undefined,
        },
      };
      const data = await predictRisk(req);
      setResult(data);
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  const donorLabels     = LOCI.map((l) => `D-${l}`);
  const recipientLabels = LOCI.map((l) => `R-${l}`);

  return (
    <div className="min-h-screen bg-[#F8F9FC]">
      {/* ── Page header ──────────────────────────────────────────────── */}
      <div className="bg-white border-b border-border">
        <div className="container max-w-7xl px-6 py-8">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div className="flex items-start gap-4">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-navy text-white shrink-0">
                <FlaskConical className="h-5 w-5" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-navy">Transplant Risk Prediction</h1>
                <p className="text-sm text-muted-foreground mt-0.5 max-w-xl">
                  Enter donor and recipient HLA alleles to generate competing-risk
                  cumulative incidence curves for GvHD, relapse, and TRM.
                </p>
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={fillExample}
              className="gap-2 shrink-0 border-blush/40 text-blush-700 hover:bg-blush-50"
            >
              <Sparkles className="h-3.5 w-3.5" />
              Try Example
            </Button>
          </div>
        </div>
      </div>

      {/* ── Demo-mode banner ─────────────────────────────────────────── */}
      {isDemoMode ? (
        <div className="bg-amber-50 border-b border-amber-200">
          <div className="container max-w-7xl px-6 py-2.5 flex items-center gap-2.5 text-sm text-amber-800">
            <WifiOff className="h-4 w-4 shrink-0 text-amber-600" />
            <span>
              <strong>Demo mode</strong> — the prediction backend is offline.
              Results shown are illustrative mock values and do not reflect real model output.
            </span>
          </div>
        </div>
      ) : null}

      {/* ── Main layout ──────────────────────────────────────────────── */}
      <div className="container max-w-7xl px-6 py-8">
        <form onSubmit={handleSubmit}>
          <div className="grid gap-8 xl:grid-cols-[560px_1fr]">

            {/* ── LEFT: form ────────────────────────────────────────── */}
            <div className="space-y-5">

              {/* HLA typing card */}
              <Card className="border-border/70 shadow-sm">
                <CardHeader className="pb-4 border-b border-border/60">
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-navy text-base">HLA Typing</CardTitle>
                      <CardDescription className="mt-0.5">
                        Standard IPD-IMGT/HLA notation, e.g.{" "}
                        <code className="code-pill">A*02:01</code>. Type to filter suggestions.
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="pt-5">
                  <div className="grid sm:grid-cols-2 gap-6">
                    <HLAFieldGroup
                      side="donor"
                      values={donorHla}
                      onChange={(l, v) => setDonorHla((p) => ({ ...p, [l]: v }))}
                    />
                    <HLAFieldGroup
                      side="recipient"
                      values={recipientHla}
                      onChange={(l, v) => setRecipientHla((p) => ({ ...p, [l]: v }))}
                    />
                  </div>

                  {/* Mismatch summary */}
                  <div className="mt-5">
                    <MismatchSummary donor={donorHla} recipient={recipientHla} />
                  </div>
                </CardContent>
              </Card>

              {/* Clinical covariates — collapsible */}
              <Card className="border-border/70 shadow-sm overflow-hidden">
                <button
                  type="button"
                  className="w-full flex items-center justify-between px-6 py-4 text-left hover:bg-muted/30 transition-colors"
                  onClick={() => setShowClinical((v) => !v)}
                >
                  <div>
                    <span className="font-semibold text-navy text-sm">Clinical Covariates</span>
                    <span className="text-muted-foreground font-normal text-sm"> — optional</span>
                  </div>
                  <motion.div
                    animate={{ rotate: showClinical ? 180 : 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    <ChevronDown className="h-4 w-4 text-muted-foreground" />
                  </motion.div>
                </button>

                <AnimatePresence initial={false}>
                  {showClinical && (
                    <motion.div
                      key="clinical"
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.25, ease }}
                      className="overflow-hidden"
                    >
                      <CardContent className="pt-2 pb-5 border-t border-border/60">
                        <div className="grid grid-cols-2 gap-4 mt-2">
                          {/* Ages */}
                          <div>
                            <Label className="text-xs text-muted-foreground mb-1.5 block">
                              Recipient age (years)
                            </Label>
                            <Input
                              type="number" min={0} max={99} placeholder="e.g. 8"
                              value={ageRecipient}
                              onChange={(e) => setAgeRecipient(e.target.value)}
                            />
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground mb-1.5 block">
                              Donor age (years)
                            </Label>
                            <Input
                              type="number" min={0} max={99} placeholder="e.g. 35"
                              value={ageDonor}
                              onChange={(e) => setAgeDonor(e.target.value)}
                            />
                          </div>

                          {/* Disease */}
                          <div>
                            <Label className="text-xs text-muted-foreground mb-1.5 block">
                              Disease type
                            </Label>
                            <Select value={disease} onValueChange={setDisease}>
                              <SelectTrigger><SelectValue placeholder="Select…" /></SelectTrigger>
                              <SelectContent>
                                <SelectItem value="ALL">ALL (acute lymphoblastic)</SelectItem>
                                <SelectItem value="AML">AML (acute myeloid)</SelectItem>
                                <SelectItem value="CML">CML (chronic myeloid)</SelectItem>
                                <SelectItem value="MDS">MDS (myelodysplastic)</SelectItem>
                                <SelectItem value="NHL">NHL (non-Hodgkin lymphoma)</SelectItem>
                                <SelectItem value="AA">AA (aplastic anaemia)</SelectItem>
                                <SelectItem value="other">Other / unspecified</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>

                          {/* Conditioning */}
                          <div>
                            <Label className="text-xs text-muted-foreground mb-1.5 block">
                              Conditioning regimen
                            </Label>
                            <Select value={conditioning} onValueChange={setConditioning}>
                              <SelectTrigger><SelectValue placeholder="Select…" /></SelectTrigger>
                              <SelectContent>
                                <SelectItem value="myeloablative">Myeloablative (MAC)</SelectItem>
                                <SelectItem value="reduced">Reduced intensity (RIC)</SelectItem>
                                <SelectItem value="nonmyeloablative">Non-myeloablative (NMA)</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>

                          {/* Stem cell source */}
                          <div>
                            <Label className="text-xs text-muted-foreground mb-1.5 block">
                              Stem cell source
                            </Label>
                            <Select value={stemCellSrc} onValueChange={setStemCellSrc}>
                              <SelectTrigger><SelectValue placeholder="Select…" /></SelectTrigger>
                              <SelectContent>
                                <SelectItem value="peripheral_blood">Peripheral blood (PBSC)</SelectItem>
                                <SelectItem value="bone_marrow">Bone marrow</SelectItem>
                                <SelectItem value="cord_blood">Cord blood</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>

                          {/* Donor type */}
                          <div>
                            <Label className="text-xs text-muted-foreground mb-1.5 block">
                              Donor type
                            </Label>
                            <Select value={donorType} onValueChange={setDonorType}>
                              <SelectTrigger><SelectValue placeholder="Select…" /></SelectTrigger>
                              <SelectContent>
                                <SelectItem value="matched_related">Matched related (MRD)</SelectItem>
                                <SelectItem value="matched_unrelated">Matched unrelated (MUD)</SelectItem>
                                <SelectItem value="mismatched_unrelated">Mismatched unrelated (MMUD)</SelectItem>
                                <SelectItem value="haplo">Haploidentical</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>

                          {/* CD34 dose */}
                          <div className="col-span-2">
                            <Label className="text-xs text-muted-foreground mb-1.5 block">
                              CD34⁺ cell dose (×10⁶/kg recipient)
                            </Label>
                            <Input
                              type="number" min={0} step={0.1} placeholder="e.g. 5.2"
                              value={cd34Dose}
                              onChange={(e) => setCd34Dose(e.target.value)}
                            />
                          </div>
                        </div>
                      </CardContent>
                    </motion.div>
                  )}
                </AnimatePresence>
              </Card>

              {/* Actions */}
              <div className="flex gap-3">
                <Button
                  type="submit"
                  size="lg"
                  variant="blush"
                  disabled={loading}
                  className="flex-1 gap-2"
                >
                  {loading ? (
                    <><Loader2 className="h-4 w-4 animate-spin" />Running inference…</>
                  ) : (
                    <><FlaskConical className="h-4 w-4" />Predict Risk</>
                  )}
                </Button>
                <Button
                  type="button"
                  size="lg"
                  variant="outline"
                  onClick={clearAll}
                  className="gap-2"
                >
                  <RotateCcw className="h-4 w-4" />
                  Clear
                </Button>
              </div>

              {/* Error */}
              <AnimatePresence>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    className="flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700"
                  >
                    <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
                    <span>{error}</span>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Research disclaimer */}
              <div className="flex gap-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-700">
                <Info className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                <p>
                  <span className="font-semibold">Research prototype only.</span>{" "}
                  CAPA is trained on 187 paediatric patients (UCI BMT).
                  Results are not validated for clinical use.
                </p>
              </div>
            </div>

            {/* ── RIGHT: results ────────────────────────────────────── */}
            <div ref={resultsRef}>
              <AnimatePresence mode="wait">
                {!result ? (
                  /* Info panel */
                  <motion.div
                    key="info"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <Card className="border-dashed border-border/60">
                      <CardHeader>
                        <CardTitle className="text-navy text-base">What you&apos;ll receive</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-5">
                        {[
                          { n: "1", title: "Risk score cards", desc: "Low / Medium / High classification for GvHD, relapse, and TRM with exact percentages at day 100, 365, and 730." },
                          { n: "2", title: "CIF curves",       desc: "Cumulative incidence functions over 730 days, one curve per competing event." },
                          { n: "3", title: "Attention heatmap", desc: "Cross-attention weights showing which donor–recipient locus pairs drive the model's prediction." },
                        ].map(({ n, title, desc }) => (
                          <div key={n} className="flex gap-3">
                            <div className="flex h-7 w-7 items-center justify-center rounded-full bg-navy/8 shrink-0">
                              <span className="text-xs font-bold text-navy">{n}</span>
                            </div>
                            <div>
                              <p className="text-sm font-medium text-foreground">{title}</p>
                              <p className="text-sm text-muted-foreground mt-0.5">{desc}</p>
                            </div>
                          </div>
                        ))}

                        <div className="pt-2 text-center">
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={fillExample}
                            className="gap-2 border-blush/40 text-blush-700 hover:bg-blush-50"
                          >
                            <Sparkles className="h-3.5 w-3.5" />
                            Load example case
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                ) : (
                  /* Results */
                  <motion.div
                    key="results"
                    initial={{ opacity: 0, y: 16 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.45, ease }}
                    className="space-y-5"
                  >
                    {/* Result header */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="h-4 w-4 text-green-500" />
                        <span className="text-sm font-medium text-navy">Prediction complete</span>
                        {result.model_version && (
                          <span className="code-pill text-muted-foreground">{result.model_version}</span>
                        )}
                      </div>
                      <Button type="button" variant="ghost" size="sm" className="gap-1.5 text-muted-foreground" disabled>
                        <Download className="h-3.5 w-3.5" />
                        Export
                      </Button>
                    </div>

                    {/* Risk cards */}
                    <RiskSummaryCards result={result} />

                    {/* Tabs: CIF + Attention + JSON */}
                    <Tabs defaultValue="curves">
                      <TabsList className="w-full">
                        <TabsTrigger value="curves"    className="flex-1 text-xs">CIF Curves</TabsTrigger>
                        <TabsTrigger value="attention" className="flex-1 text-xs">Attention</TabsTrigger>
                        <TabsTrigger value="json"      className="flex-1 text-xs">JSON</TabsTrigger>
                      </TabsList>

                      {/* CIF chart */}
                      <TabsContent value="curves">
                        <Card className="border-border/60 shadow-sm">
                          <CardHeader className="pb-2">
                            <CardTitle className="text-navy text-sm">
                              Cumulative Incidence Functions
                            </CardTitle>
                            <CardDescription>
                              Predicted probability of each event by day post-transplant.
                              Reference lines at day 100, 365, and 730.
                            </CardDescription>
                          </CardHeader>
                          <CardContent>
                            <RiskChart
                              gvhd={result.gvhd.cumulative_incidence}
                              relapse={result.relapse.cumulative_incidence}
                              trm={result.trm.cumulative_incidence}
                              timeBins={result.gvhd.cumulative_incidence.length}
                            />
                            {/* Legend detail */}
                            <div className="mt-4 grid grid-cols-3 gap-3">
                              {([
                                { key: "gvhd",    color: "#E69F00" },
                                { key: "relapse", color: "#0072B2" },
                                { key: "trm",     color: "#D55E00" },
                              ] as const).map(({ key, color }) => {
                                const cfg = EVENT_CONFIG[key];
                                const cif = result[key].cumulative_incidence;
                                const N   = cif.length;
                                const last = (cif[N - 1] * 100).toFixed(1);
                                return (
                                  <div key={key} className="flex items-center gap-2 text-xs">
                                    <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
                                    <span className="font-medium">{cfg.label}</span>
                                    <span className="text-muted-foreground ml-auto">{last}%</span>
                                  </div>
                                );
                              })}
                            </div>
                          </CardContent>
                        </Card>
                      </TabsContent>

                      {/* Attention heatmap */}
                      <TabsContent value="attention">
                        <Card className="border-border/60 shadow-sm">
                          <CardHeader className="pb-2">
                            <CardTitle className="text-navy text-sm">
                              Cross-Attention Weights
                            </CardTitle>
                            <CardDescription>
                              Donor loci (rows) × Recipient loci (columns).
                              Darker = higher attention weight.
                            </CardDescription>
                          </CardHeader>
                          <CardContent>
                            {result.attention_weights ? (
                              <AttentionHeatmap
                                weights={result.attention_weights}
                                donorLabels={donorLabels}
                                recipientLabels={recipientLabels}
                              />
                            ) : (
                              <p className="text-sm text-muted-foreground py-4 text-center">
                                Attention weights not available for this prediction.
                              </p>
                            )}
                          </CardContent>
                        </Card>
                      </TabsContent>

                      {/* Raw JSON */}
                      <TabsContent value="json">
                        <Card className="border-border/60 shadow-sm">
                          <CardHeader className="pb-2">
                            <CardTitle className="text-navy text-sm">Raw API Response</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <pre className="text-xs bg-muted rounded-lg p-4 overflow-auto max-h-[480px] scrollbar-thin leading-relaxed">
                              {JSON.stringify(result, null, 2)}
                            </pre>
                          </CardContent>
                        </Card>
                      </TabsContent>
                    </Tabs>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
