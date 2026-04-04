import Link from "next/link";
import { BookOpen, ArrowLeft, FileText } from "lucide-react";
import { GithubIcon } from "@/components/ui/github-icon";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const SECTIONS = [
  { n: "1", title: "Introduction",          desc: "Motivation, HLA biology background, and gap in the literature." },
  { n: "2", title: "Methods",               desc: "ESM-2 encoding, cross-attention interaction network, DeepHit competing-risks head, UCI BMT dataset." },
  { n: "3", title: "Experiments",           desc: "C-index, Brier score, calibration — comparison with Cox-PH, RSF, and DeepSurv baselines." },
  { n: "4", title: "Interpretability",      desc: "SHAP values for clinical covariates; attention heatmaps for allele-pair importance." },
  { n: "5", title: "Discussion",            desc: "Limitations (small N, paediatric cohort); future directions (larger multicenter data, HLA imputation)." },
  { n: "6", title: "Supplementary",         desc: "Hyperparameter sensitivity; UMAP of HLA embedding space; calibration scatter plots." },
];

export default function PaperPage() {
  return (
    <div className="bg-white min-h-screen">
      {/* Header */}
      <section className="hero-bg relative overflow-hidden py-20">
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
            Manuscript
          </Badge>
          <h1 className="mx-auto max-w-3xl text-4xl font-bold text-white sm:text-5xl leading-tight">
            Structure-aware HLA mismatch{" "}
            <span className="gradient-text">representations</span>{" "}
            for post-transplant outcome prediction
          </h1>
          <p className="mx-auto mt-5 max-w-xl text-white/55">
            CAPA Contributors · 2025 · Preprint
          </p>
          <div className="mt-8 flex justify-center gap-4">
            <Button variant="blush" size="lg" disabled>
              <FileText className="h-4 w-4" />
              PDF (coming soon)
            </Button>
            <Button
              asChild
              size="lg"
              className="border-white/20 bg-white/5 text-white hover:bg-white/10 hover:text-white"
              variant="outline"
            >
              <a
                href="https://github.com/capa-project/capa"
                target="_blank"
                rel="noopener noreferrer"
              >
                <GithubIcon className="h-4 w-4" />
                Code
              </a>
            </Button>
          </div>
        </div>
      </section>

      {/* Abstract */}
      <section className="py-16">
        <div className="container max-w-3xl px-6">
          <Badge variant="blush" className="mb-4">Abstract</Badge>
          <div className="prose prose-slate max-w-none">
            <p className="text-muted-foreground leading-relaxed">
              HLA compatibility between donor and recipient is the primary determinant
              of haematopoietic stem cell transplantation (HSCT) outcome, yet clinical
              practice still relies on coarse categorical match/mismatch scores that
              discard most of the immunological information encoded in allele sequences.
              We introduce CAPA (Computational Architecture for Predicting Alloimmunity),
              a framework that replaces categorical HLA encoding with continuous
              1 280-dimensional embeddings from ESM-2, a 650 M-parameter protein language
              model pre-trained on 250 million protein sequences.
            </p>
            <p className="text-muted-foreground leading-relaxed mt-4">
              A cross-attention interaction network learns which donor–recipient allele
              pairs are immunologically conflicting, producing interpretable attention
              weights alongside a 128-dim interaction feature vector. A DeepHit
              competing-risks head jointly models acute graft-versus-host disease (GvHD),
              relapse, and transplant-related mortality (TRM) as competing events,
              outputting calibrated cumulative incidence functions. Evaluated on the UCI
              Bone Marrow Transplant Children dataset (n = 187), CAPA improves time-dependent
              C-index for GvHD prediction by X.XX points over Cox-PH and by X.XX over random
              survival forests, while maintaining competitive calibration. Attention weights
              and SHAP explanations provide patient-level interpretability.
            </p>
          </div>
        </div>
      </section>

      {/* Sections overview */}
      <section className="bg-[#F8F9FC] py-16">
        <div className="container max-w-7xl px-6">
          <div className="mb-10">
            <Badge variant="blush" className="mb-4">Paper structure</Badge>
            <h2 className="text-2xl font-bold text-navy">Contents</h2>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {SECTIONS.map(({ n, title, desc }) => (
              <Card key={n} className="border-border/60 shadow-sm">
                <CardHeader className="pb-2">
                  <div className="flex items-center gap-3">
                    <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-navy text-white text-xs font-bold font-mono">
                      {n}
                    </span>
                    <CardTitle className="text-navy text-sm">{title}</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground leading-relaxed">{desc}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Citation placeholder */}
      <section className="py-16">
        <div className="container max-w-3xl px-6">
          <Badge variant="blush" className="mb-4">Citation</Badge>
          <pre className="rounded-xl bg-muted border border-border p-5 text-xs font-mono text-foreground overflow-auto scrollbar-thin leading-relaxed">
{`@article{capa2025,
  title   = {Structure-aware HLA mismatch representations
             for post-transplant outcome prediction},
  author  = {CAPA Contributors},
  year    = {2025},
  note    = {Preprint},
  url     = {https://github.com/capa-project/capa}
}`}
          </pre>
        </div>
      </section>

      {/* Back link */}
      <section className="bg-[#F8F9FC] border-t border-border py-10">
        <div className="container max-w-7xl px-6 flex gap-4">
          <Button asChild variant="outline">
            <Link href="/">
              <ArrowLeft className="h-4 w-4" />
              Back to home
            </Link>
          </Button>
          <Button asChild variant="blush">
            <Link href="/predict">
              <BookOpen className="h-4 w-4" />
              Try the prediction tool
            </Link>
          </Button>
        </div>
      </section>
    </div>
  );
}
