import Link from "next/link";
import { Database, Microscope, BookOpen, ArrowRight } from "lucide-react";
import { GithubIcon } from "@/components/ui/github-icon";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const TECH = [
  { label: "Model",     value: "ESM-2 (facebook/esm2_t33_650M_UR50D) via HuggingFace" },
  { label: "Framework", value: "PyTorch 2.x" },
  { label: "Survival",  value: "DeepHit — joint distribution over event types and times" },
  { label: "Data",      value: "UCI Bone Marrow Transplant Dataset — 187 paediatric patients" },
  { label: "HLA ref.",  value: "IPD-IMGT/HLA database (full protein sequences)" },
  { label: "Explainability", value: "SHAP (KernelExplainer) + cross-attention weights" },
  { label: "Frontend",  value: "Next.js 14, Tailwind CSS, shadcn/ui" },
  { label: "License",   value: "MIT" },
];

const LIMITATIONS = [
  "Training cohort of 187 patients is small — deep model is proof-of-concept only.",
  "Paediatric HSCT only (UCI BMT). Adult cohort generalisability is unknown.",
  "HLA sequences limited to IPD-IMGT/HLA alleles; novel or incomplete typings may fail.",
  "Not validated for clinical decision-making. Research use only.",
];

export default function AboutPage() {
  return (
    <div className="bg-white">
      {/* ── Hero ─────────────────────────────────────────────────────── */}
      <section className="hero-bg relative overflow-hidden py-24">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              "radial-gradient(ellipse 70% 50% at 50% 0%, rgba(232,132,154,0.12) 0%, transparent 70%)",
          }}
        />
        <div className="container relative max-w-7xl px-6 text-center">
          <Badge className="mb-6 border-white/15 bg-white/8 text-white/70">
            About the Project
          </Badge>
          <h1 className="mx-auto max-w-3xl text-4xl font-bold text-white sm:text-5xl">
            Rethinking HLA compatibility with{" "}
            <span className="gradient-text">protein language models</span>
          </h1>
          <p className="mx-auto mt-6 max-w-2xl text-lg text-white/60 leading-relaxed">
            CAPA replaces the coarse categorical HLA match/mismatch score with
            continuous, biologically meaningful distances derived from ESM-2 —
            a 650 M-parameter protein language model.
          </p>
        </div>
      </section>

      {/* ── Story ─────────────────────────────────────────────────────── */}
      <section className="py-20">
        <div className="container max-w-3xl px-6 prose prose-slate prose-headings:text-navy prose-headings:tracking-tight max-w-none">
          <h2 className="text-2xl font-bold text-navy mb-4">The Problem</h2>
          <p className="text-muted-foreground leading-relaxed mb-4">
            Haematopoietic stem cell transplantation (HSCT) outcome depends
            critically on HLA compatibility between donor and recipient. The
            standard approach encodes this as a binary match/mismatch count —
            treating all mismatches as equally dangerous regardless of the
            underlying protein sequence difference.
          </p>
          <p className="text-muted-foreground leading-relaxed mb-4">
            A mismatch between two structurally similar alleles and two
            completely divergent ones are treated identically. This discards
            most of the immunologically relevant information.
          </p>

          <h2 className="text-2xl font-bold text-navy mb-4 mt-12">Our Approach</h2>
          <p className="text-muted-foreground leading-relaxed mb-4">
            CAPA encodes every HLA allele as a 1 280-dimensional vector using
            ESM-2, a large protein language model pre-trained on 250 million
            protein sequences. Alleles with similar binding-groove conformations
            cluster together in embedding space; immunologically distant alleles
            are far apart.
          </p>
          <p className="text-muted-foreground leading-relaxed mb-4">
            A cross-attention network then models the <em>interaction</em>{" "}
            between donor and recipient allele sets, learning which locus pairs
            are most predictive. The resulting representation is fed into a
            DeepHit survival head that jointly models GvHD, relapse, and
            transplant-related mortality as competing risks.
          </p>

          <h2 className="text-2xl font-bold text-navy mb-4 mt-12">Data</h2>
          <p className="text-muted-foreground leading-relaxed mb-4">
            Primary validation uses the{" "}
            <a
              href="https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blush-600 hover:text-blush-700 underline underline-offset-2"
            >
              UCI Bone Marrow Transplant Children Dataset
            </a>{" "}
            — 187 paediatric patients with allogeneic HSCT, donor/recipient HLA
            typing at antigen and allele resolution, and time-to-event outcomes
            for GvHD, relapse, and cause of death. HLA protein sequences are
            sourced from the{" "}
            <a
              href="https://www.ebi.ac.uk/ipd/imgt/hla/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blush-600 hover:text-blush-700 underline underline-offset-2"
            >
              IPD-IMGT/HLA database
            </a>
            .
          </p>
        </div>
      </section>

      {/* ── Tech stack ────────────────────────────────────────────────── */}
      <section className="bg-[#F8F9FC] py-20">
        <div className="container max-w-7xl px-6">
          <div className="mb-12">
            <Badge variant="blush" className="mb-4">Technical details</Badge>
            <h2 className="text-2xl font-bold text-navy">Stack &amp; references</h2>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            {TECH.map(({ label, value }) => (
              <div
                key={label}
                className="flex items-start gap-4 rounded-xl bg-white border border-border/60 p-4 shadow-sm"
              >
                <span className="text-xs font-semibold text-navy/60 uppercase tracking-wider w-28 shrink-0 pt-0.5">
                  {label}
                </span>
                <span className="text-sm text-foreground">{value}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Limitations ───────────────────────────────────────────────── */}
      <section className="py-20">
        <div className="container max-w-7xl px-6">
          <div className="mb-12">
            <Badge variant="blush" className="mb-4">Transparency</Badge>
            <h2 className="text-2xl font-bold text-navy">Limitations</h2>
            <p className="text-muted-foreground mt-2">
              We are committed to honest reporting of what CAPA can and cannot do.
            </p>
          </div>

          <Card className="border-amber-200 bg-amber-50">
            <CardHeader>
              <CardTitle className="text-amber-800 text-base flex items-center gap-2">
                <Microscope className="h-4 w-4" />
                Research prototype — not for clinical use
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {LIMITATIONS.map((lim) => (
                  <li key={lim} className="flex items-start gap-2 text-sm text-amber-800">
                    <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-amber-500 shrink-0" />
                    {lim}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* ── CTA ───────────────────────────────────────────────────────── */}
      <section className="bg-[#F8F9FC] py-20 border-t border-border">
        <div className="container max-w-7xl px-6">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6">
            <div>
              <h2 className="text-xl font-bold text-navy">
                Interested in contributing?
              </h2>
              <p className="text-muted-foreground mt-1">
                CAPA is open source. Pull requests, issues, and feedback are welcome.
              </p>
            </div>
            <div className="flex gap-3">
              <Button asChild variant="default">
                <a
                  href="https://github.com/capa-project/capa"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <GithubIcon className="h-4 w-4" />
                  GitHub
                </a>
              </Button>
              <Button asChild variant="blush">
                <Link href="/predict">
                  Try the tool
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
