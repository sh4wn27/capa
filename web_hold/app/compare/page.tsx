"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Users,
  Plus,
  Trash2,
  Loader2,
  Trophy,
  WifiOff,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Sparkles,
} from "lucide-react";
import { Button }   from "@/components/ui/button";
import { Input }    from "@/components/ui/input";
import { Label }    from "@/components/ui/label";
import { Badge }    from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { HLACombobox } from "@/components/HLACombobox";
import { comparedonors, type ComparisonRequest, type DonorRiskSummary } from "@/lib/api";
import { cn } from "@/lib/utils";

/* ─── constants ─────────────────────────────────────────────────────── */
const LOCI = ["A", "B", "C", "DRB1", "DQB1", "DPB1"] as const;
type Locus = (typeof LOCI)[number];

/* ─── mock comparison data for demo mode ─────────────────────────────── */
function buildMockComparison(donors: { label: string }[]) {
  const risks = donors.map((d, i) => ({
    label: d.label || `Donor ${i + 1}`,
    gvhd_risk:    Number((0.15 + i * 0.08 + Math.random() * 0.05).toFixed(3)),
    relapse_risk: Number((0.20 + i * 0.04 + Math.random() * 0.05).toFixed(3)),
    trm_risk:     Number((0.12 + i * 0.06 + Math.random() * 0.04).toFixed(3)),
    mismatch_count: i,
    rank: 0,
    full_prediction: null as never,
  }));
  risks.sort((a, b) => (a.gvhd_risk + a.trm_risk) - (b.gvhd_risk + b.trm_risk));
  risks.forEach((r, i) => { r.rank = i + 1; });
  return { donors: risks, best_donor_label: risks[0].label, model_version: "demo" };
}

/* ─── risk bar ───────────────────────────────────────────────────────── */
function RiskBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-black/8 overflow-hidden">
        <motion.div
          className={cn("h-full rounded-full", color)}
          initial={{ width: 0 }}
          animate={{ width: `${Math.round(value * 100)}%` }}
          transition={{ duration: 0.6, ease: [0.21, 0.47, 0.32, 0.98] }}
        />
      </div>
      <span className="text-xs tabular-nums w-10 text-right text-muted-foreground">
        {Math.round(value * 100)}%
      </span>
    </div>
  );
}

/* ─── donor result row ───────────────────────────────────────────────── */
function DonorRow({
  summary,
  isExpanded,
  onToggle,
}: {
  summary: DonorRiskSummary;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const isBest = summary.rank === 1;

  return (
    <div
      className={cn(
        "rounded-xl border transition-colors",
        isBest ? "border-emerald-300 bg-emerald-50/60" : "border-border bg-white",
      )}
    >
      {/* Summary row */}
      <button
        className="w-full flex items-center gap-4 p-4 text-left"
        onClick={onToggle}
        type="button"
      >
        {/* Rank badge */}
        <div
          className={cn(
            "flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-bold",
            isBest
              ? "bg-emerald-500 text-white"
              : "bg-muted text-muted-foreground",
          )}
        >
          {isBest ? <Trophy className="h-4 w-4" /> : summary.rank}
        </div>

        {/* Label + mismatch */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm truncate">{summary.label}</span>
            {isBest && (
              <Badge className="text-[10px] bg-emerald-100 text-emerald-700 border-emerald-200 border">
                Best match
              </Badge>
            )}
          </div>
          {summary.mismatch_count != null && (
            <p className="text-xs text-muted-foreground mt-0.5">
              {summary.mismatch_count} locus mismatch{summary.mismatch_count !== 1 ? "es" : ""}
            </p>
          )}
        </div>

        {/* Risk columns */}
        <div className="hidden sm:grid grid-cols-3 gap-6 w-72 shrink-0">
          <div>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">GvHD</p>
            <RiskBar value={summary.gvhd_risk} color="bg-amber-400" />
          </div>
          <div>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Relapse</p>
            <RiskBar value={summary.relapse_risk} color="bg-blue-500" />
          </div>
          <div>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">TRM</p>
            <RiskBar value={summary.trm_risk} color="bg-orange-600" />
          </div>
        </div>

        {/* Expand toggle */}
        {summary.full_prediction ? (
          isExpanded
            ? <ChevronUp className="h-4 w-4 shrink-0 text-muted-foreground" />
            : <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground" />
        ) : null}
      </button>

      {/* Mobile risk bars */}
      <div className="sm:hidden grid grid-cols-3 gap-3 px-4 pb-3">
        {(["GvHD", "Relapse", "TRM"] as const).map((label, i) => {
          const val = [summary.gvhd_risk, summary.relapse_risk, summary.trm_risk][i];
          const color = ["bg-amber-400", "bg-blue-500", "bg-orange-600"][i];
          return (
            <div key={label}>
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">{label}</p>
              <RiskBar value={val} color={color} />
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ─── donor input card ───────────────────────────────────────────────── */
function DonorInput({
  index,
  label,
  hla,
  onLabelChange,
  onHlaChange,
  onRemove,
  canRemove,
}: {
  index: number;
  label: string;
  hla: Record<string, string>;
  onLabelChange: (v: string) => void;
  onHlaChange: (locus: string, v: string) => void;
  onRemove: () => void;
  canRemove: boolean;
}) {
  return (
    <Card className="relative">
      <CardHeader className="pb-3 pt-4 px-4">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-navy text-white text-xs font-bold">
              {index + 1}
            </div>
            <Input
              value={label}
              onChange={(e) => onLabelChange(e.target.value)}
              placeholder={`Donor ${index + 1}`}
              className="h-7 text-sm font-medium border-0 px-1 focus-visible:ring-0 bg-transparent"
            />
          </div>
          {canRemove && (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 shrink-0 text-muted-foreground hover:text-destructive"
              onClick={onRemove}
              type="button"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="px-4 pb-4 pt-0">
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
          {LOCI.map((locus) => (
            <div key={locus}>
              <Label className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1 block">
                {locus}
              </Label>
              <HLACombobox
                locus={locus}
                value={hla[locus] ?? ""}
                onChange={(v) => onHlaChange(locus, v)}
              />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

/* ─── page ──────────────────────────────────────────────────────────── */
export default function ComparePage() {
  /* Recipient */
  const [recipientHla, setRecipientHla] = useState<Record<string, string>>({});

  /* Donors */
  type DonorState = { label: string; hla: Record<string, string> };
  const [donors, setDonors] = useState<DonorState[]>([
    { label: "Donor A", hla: {} },
    { label: "Donor B", hla: {} },
  ]);

  /* Result */
  type CompResult = ReturnType<typeof buildMockComparison>;
  const [result, setResult] = useState<CompResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  /* Demo mode */
  const [isDemoMode, setIsDemoMode] = useState(false);
  useEffect(() => {
    let cancelled = false;
    fetch("/api/predict", { method: "GET" })
      .then((r) => { if (!cancelled) setIsDemoMode(!r.ok); })
      .catch(() => { if (!cancelled) setIsDemoMode(true); });
    return () => { cancelled = true; };
  }, []);

  function addDonor() {
    if (donors.length >= 20) return;
    setDonors((prev) => [...prev, { label: `Donor ${String.fromCharCode(65 + prev.length)}`, hla: {} }]);
  }

  function removeDonor(i: number) {
    setDonors((prev) => prev.filter((_, idx) => idx !== i));
  }

  function updateDonorLabel(i: number, label: string) {
    setDonors((prev) => prev.map((d, idx) => idx === i ? { ...d, label } : d));
  }

  function updateDonorHla(i: number, locus: string, value: string) {
    setDonors((prev) =>
      prev.map((d, idx) =>
        idx === i ? { ...d, hla: { ...d.hla, [locus]: value } } : d,
      ),
    );
  }

  function fillExample() {
    setRecipientHla({ A: "A*03:01", B: "B*08:01", C: "C*07:01", DRB1: "DRB1*03:01", DQB1: "DQB1*02:01" });
    setDonors([
      { label: "Donor A (10/10)",  hla: { A: "A*03:01", B: "B*08:01", C: "C*07:01", DRB1: "DRB1*03:01", DQB1: "DQB1*02:01" } },
      { label: "Donor B (9/10)",   hla: { A: "A*02:01", B: "B*08:01", C: "C*07:01", DRB1: "DRB1*03:01", DQB1: "DQB1*02:01" } },
      { label: "Donor C (8/10)",   hla: { A: "A*02:01", B: "B*07:02", C: "C*07:02", DRB1: "DRB1*03:01", DQB1: "DQB1*02:01" } },
    ]);
    setError(null);
    setResult(buildMockComparison([
      { label: "Donor A (10/10)" },
      { label: "Donor B (9/10)" },
      { label: "Donor C (8/10)" },
    ]));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      if (isDemoMode) {
        await new Promise((r) => setTimeout(r, 700));
        setResult(buildMockComparison(donors));
        return;
      }

      const req: ComparisonRequest = {
        recipient_hla: recipientHla,
        donors: donors.map((d) => ({ label: d.label, donor_hla: d.hla })),
      };
      const data = await comparedonors(req);
      setResult(data as CompResult);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#F8F9FC]">
      {/* ── Header ────────────────────────────────────────────────── */}
      <div className="bg-white border-b border-border">
        <div className="container max-w-5xl px-6 py-8">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div className="flex items-start gap-4">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-navy text-white shrink-0">
                <Users className="h-5 w-5" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-navy">Donor Comparison</h1>
                <p className="text-sm text-muted-foreground mt-0.5 max-w-xl">
                  Rank multiple candidate donors against one recipient by predicted
                  GvHD, relapse, and TRM risk.
                </p>
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={fillExample}
              className="gap-2 shrink-0 border-blush/40 text-blush-700 hover:bg-blush-50"
              type="button"
            >
              <Sparkles className="h-3.5 w-3.5" />
              Try Example
            </Button>
          </div>
        </div>
      </div>

      {/* ── Demo banner ───────────────────────────────────────────── */}
      {isDemoMode ? (
        <div className="bg-amber-50 border-b border-amber-200">
          <div className="container max-w-5xl px-6 py-2.5 flex items-center gap-2.5 text-sm text-amber-800">
            <WifiOff className="h-4 w-4 shrink-0 text-amber-600" />
            <span>
              <strong>Demo mode</strong> — backend offline. Results are illustrative only.
            </span>
          </div>
        </div>
      ) : null}

      <div className="container max-w-5xl px-6 py-8 space-y-8">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* ── Recipient ──────────────────────────────────────────── */}
          <section>
            <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">
              Recipient HLA
            </h2>
            <Card>
              <CardContent className="p-4">
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
                  {LOCI.map((locus) => (
                    <div key={locus}>
                      <Label className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1 block">
                        {locus}
                      </Label>
                      <HLACombobox
                        locus={locus}
                        value={recipientHla[locus] ?? ""}
                        onChange={(v) =>
                          setRecipientHla((prev) => ({ ...prev, [locus]: v }))
                        }
                      />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </section>

          {/* ── Donors ─────────────────────────────────────────────── */}
          <section>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
                Candidate Donors ({donors.length}/20)
              </h2>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={addDonor}
                disabled={donors.length >= 20}
                className="gap-1.5 h-7 text-xs"
              >
                <Plus className="h-3 w-3" />
                Add donor
              </Button>
            </div>
            <div className="space-y-3">
              {donors.map((donor, i) => (
                <DonorInput
                  key={i}
                  index={i}
                  label={donor.label}
                  hla={donor.hla}
                  onLabelChange={(v) => updateDonorLabel(i, v)}
                  onHlaChange={(locus, v) => updateDonorHla(i, locus, v)}
                  onRemove={() => removeDonor(i)}
                  canRemove={donors.length > 2}
                />
              ))}
            </div>
          </section>

          {/* ── Submit ─────────────────────────────────────────────── */}
          {error ? (
            <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/5 border border-destructive/20 rounded-lg px-4 py-3">
              <AlertCircle className="h-4 w-4 shrink-0" />
              {error}
            </div>
          ) : null}

          <Button
            type="submit"
            disabled={loading || donors.length < 2}
            className="w-full bg-navy hover:bg-navy/90 text-white h-11"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Comparing donors…
              </>
            ) : (
              `Compare ${donors.length} donors`
            )}
          </Button>
        </form>

        {/* ── Results ────────────────────────────────────────────────── */}
        <AnimatePresence>
          {result ? (
            <motion.section
              key="results"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
                  Ranked Results
                </h2>
                <Badge variant="outline" className="text-xs">
                  Best: {result.best_donor_label}
                </Badge>
              </div>

              <div className="space-y-2">
                {result.donors.map((summary, i) => (
                  <DonorRow
                    key={summary.label}
                    summary={summary as DonorRiskSummary}
                    isExpanded={expandedIdx === i}
                    onToggle={() => setExpandedIdx(expandedIdx === i ? null : i)}
                  />
                ))}
              </div>

              <p className="text-xs text-muted-foreground mt-4">
                Ranked by composite GvHD + TRM risk (lower = better match).
                Relapse risk is shown for reference and does not affect ranking.
                {isDemoMode && " Results are illustrative mock values."}
              </p>
            </motion.section>
          ) : null}
        </AnimatePresence>
      </div>
    </div>
  );
}
