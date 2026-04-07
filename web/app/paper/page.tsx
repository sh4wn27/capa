import Link from "next/link";
import { BookOpen, ArrowLeft, FileText } from "lucide-react";
import { GithubIcon } from "@/components/ui/github-icon";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const SECTIONS = [
  { n: "1", title: "Introduction",     desc: "Motivation, competing-risks framing, HLA biology background, and gap in the literature." },
  { n: "2", title: "Methods",          desc: "ESM-2 encoding, cross-attention interaction network, DeepHit competing-risks head, UCI BMT dataset description." },
  { n: "3", title: "Results",          desc: "C-index and Brier score benchmarks — Fine–Gray, Cox-PH, and flat-feature DeepHit on the held-out test set (n = 29)." },
  { n: "4", title: "Discussion",       desc: "Honest framing of CAPA as proposed architecture; limitations (small N, paediatric cohort, no allele-level HLA); future directions." },
  { n: "5", title: "Supplementary",    desc: "Full cohort table (n = 187), IBS results, compute/parameter breakdown, hyperparameter sensitivity analysis." },
];

export default function PaperPage() {
  return (
    <div className="bg-white min-h-screen">
      {/* Header */}
      <section className="relative overflow-hidden py-20 bg-white border-b border-border">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              "radial-gradient(ellipse 70% 50% at 50% 0%, rgba(232,132,154,0.12) 0%, transparent 70%)",
          }}
        />
        <div className="container relative max-w-7xl px-6 text-center">
          <Badge className="mb-6" variant="blush">
            Manuscript
          </Badge>
          <h1 className="mx-auto max-w-3xl text-4xl font-bold text-foreground sm:text-5xl leading-tight">
            Structure-aware HLA mismatch{" "}
            <span className="gradient-text">representations</span>{" "}
            for post-transplant outcome prediction
          </h1>
          <p className="mx-auto mt-5 max-w-xl text-black/50">
            Huanxuan Li (Shawn) · 2025 · Preprint
          </p>
          <div className="mt-8 flex justify-center gap-4">
            <Button asChild variant="blush" size="lg">
              <a href="/capa_manuscript.pdf" target="_blank" rel="noopener noreferrer">
                <FileText className="h-4 w-4" />
                Read PDF
              </a>
            </Button>
            <Button
              asChild
              size="lg"
              variant="outline"
              className="border-black/15 text-foreground hover:bg-black/5"
            >
              <a
                href="https://github.com/sh4wn27/capa"
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
              <strong className="text-foreground">Motivation:</strong>{" "}
              Donor selection for allogeneic haematopoietic stem cell transplantation
              (HSCT) relies on categorical HLA match/mismatch counts that treat
              antigenically distinct alleles as equivalent once they share a mismatch
              flag. This representation discards the amino-acid-level structural
              divergence between allele pairs and ignores the competing nature of
              post-transplant events: graft-versus-host disease (GvHD), relapse, and
              transplant-related mortality (TRM) preclude one another and must be
              modelled jointly.
            </p>
            <p className="text-muted-foreground leading-relaxed mt-4">
              <strong className="text-foreground">Methods:</strong>{" "}
              We introduce CAPA (Computational Architecture for Predicting Alloimmunity),
              a deep learning framework that encodes each HLA allele as a 1 280-dimensional
              vector using the frozen ESM-2 650M protein language model. Donor–recipient
              interaction features are extracted by a bidirectional cross-attention network
              (2 layers, 8 heads, d′=128), concatenated with clinical covariates and passed
              to a DeepHit head that jointly estimates the discrete-time cumulative incidence
              functions for all three competing events over a 730-day horizon. Only the
              interaction network and survival head (~2.8M parameters) are trained; ESM-2
              remains frozen.
            </p>
            <p className="text-muted-foreground leading-relaxed mt-4">
              <strong className="text-foreground">Results:</strong>{" "}
              On the public UCI Bone Marrow Transplant dataset (n = 187 paediatric patients,
              train/val/test 70/15/15%), we benchmark tabular-feature competing-risks models
              as reference comparators for CAPA. The Fine–Gray subdistribution hazard model
              achieves concordance indices of 0.84 (95% CI 0.69–1.00) for relapse and 0.66
              (0.48–0.86) for TRM on the held-out test set (n = 29); cause-specific Cox
              reaches 0.75 (0.53–1.00) and 0.65 (0.46–0.85). A flat-feature DeepHit MLP
              performs below the classical baselines (relapse 0.67, TRM 0.41), consistent
              with the known difficulty of training deep survival models on small cohorts.
              The GvHD endpoint could not be evaluated reliably (only n = 2 GvHD events in
              the test set). Full validation of CAPA's ESM-2 pipeline requires a registry
              dataset with allele-level HLA typing, which we identify as the primary
              direction for future work.
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
  author  = {Li, Huanxuan},
  year    = {2025},
  note    = {Preprint},
  url     = {https://github.com/sh4wn27/capa}
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
