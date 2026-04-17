import Link from "next/link";
import { ExternalLink, Database, Microscope, ArrowRight, FileText } from "lucide-react";
import { GithubIcon } from "@/components/ui/github-icon";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const DATASET_FEATURES = [
  {
    group: "Recipient",
    items: [
      "Age at transplant",
      "Body weight",
      "Gender",
      "Disease type (ALL, AML, chronic leukemia, lymphoma, etc.)",
      "Disease stage (early / intermediate / advanced)",
      "CMV serostatus",
      "HLA-A, B, C, DRB1, DQB1 — antigen and allele level",
    ],
  },
  {
    group: "Donor",
    items: [
      "Age",
      "Gender",
      "Donor–recipient gender match",
      "CMV serostatus",
      "Relation to recipient (sibling, unrelated, etc.)",
      "HLA-A, B, C, DRB1, DQB1 — antigen and allele level",
    ],
  },
  {
    group: "Transplant",
    items: [
      "Stem cell source (bone marrow / peripheral blood / cord blood)",
      "CD34+ cell dose",
      "CD3+ cell dose",
      "ABO blood group compatibility",
      "Conditioning regimen intensity",
      "T-cell depletion",
    ],
  },
  {
    group: "Outcomes",
    items: [
      "Overall survival time (days)",
      "Survival status (alive / dead)",
      "Acute GvHD onset and grade",
      "Chronic GvHD presence",
      "Relapse (time + indicator)",
      "Cause of death (GvHD / relapse / TRM / other)",
    ],
  },
];

const MODEL_RESULTS = [
  { model: "Cox-PH (cause-specific)", relapse: "0.75", trm: "0.65", note: "Baseline — tabular HLA antigen counts" },
  { model: "Fine–Gray",               relapse: "0.84", trm: "0.66", note: "Best baseline — handles competing risks directly", highlight: true },
  { model: "DeepHit (tabular HLA)",   relapse: "0.67", trm: "0.41", note: "Deep model underfit on 187 patients — too few samples" },
];

const TECH = [
  { label: "Model",          value: "ESM-2 650M (facebook/esm2_t33_650M_UR50D) via HuggingFace Transformers" },
  { label: "Framework",      value: "PyTorch 2.x, frozen encoder + trainable interaction head" },
  { label: "Survival head",  value: "DeepHit — joint discrete-time distribution over event types and times" },
  { label: "Interaction",    value: "Multi-head cross-attention (donor queries, recipient keys/values)" },
  { label: "Explainability", value: "SHAP (KernelExplainer) for covariates + cross-attention weight maps" },
  { label: "Data",           value: "UCI BMT Children Dataset — 187 paediatric HSCT patients" },
  { label: "HLA sequences",  value: "IPD-IMGT/HLA database — full protein sequences per allele" },
  { label: "Embedding cache",value: "HDF5 on disk — ESM-2 inference runs once, embeddings reused" },
  { label: "Frontend",       value: "Next.js 14, Tailwind CSS, shadcn/ui, Framer Motion" },
  { label: "License",        value: "MIT" },
];

const LIMITATIONS = [
  "187 patients is too small to reliably train a deep model. Fine–Gray outperforms DeepHit here — this is expected and honest.",
  "Paediatric HSCT only. Adult cohort generalisability is untested.",
  "HLA sequences are limited to alleles present in IPD-IMGT/HLA. Incomplete or novel typings fail gracefully but lose embedding fidelity.",
  "ESM-2 was pre-trained on general protein sequences, not HLA-specific data. Fine-tuning on immunological corpora could improve representations substantially.",
  "Competing-risks calibration is not validated externally. The Brier score numbers should be treated as within-dataset estimates.",
  "Not validated for clinical decision-making. Research and educational use only.",
];

export default function AboutPage() {
  return (
    <div className="bg-white">

      {/* ── Hero ─────────────────────────────────────────────────────── */}
      <section className="py-24 bg-white border-b border-border">
        <div className="container max-w-3xl px-6">
          <Badge className="mb-6" variant="blush">About the Project</Badge>
          <h1 className="text-4xl sm:text-5xl font-semibold text-foreground leading-tight">
            Rethinking HLA compatibility
            <br />
            with <span className="gradient-text">protein language models</span>
          </h1>
          <p className="mt-6 text-lg text-black/50 leading-relaxed">
            CAPA is an open-source research framework that replaces the binary HLA
            match/mismatch score with continuous, structurally-informed distances —
            then uses them to predict post-transplant outcomes as competing risks.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Button asChild variant="blush">
              <a
                href="https://docs.google.com/document/d/1IALiVoo6vYtWGkGcxmqoMyF_gyQNMmClsotXd8b5JxQ/edit?usp=sharing"
                target="_blank"
                rel="noopener noreferrer"
              >
                <FileText className="h-4 w-4" />
                Technical Debrief (v1)
                <ExternalLink className="h-3.5 w-3.5 opacity-60" />
              </a>
            </Button>
            <Button asChild variant="outline">
              <a href="https://github.com/sh4wn27/capa" target="_blank" rel="noopener noreferrer">
                <GithubIcon className="h-4 w-4" />
                GitHub
              </a>
            </Button>
          </div>
        </div>
      </section>

      {/* ── Origin story ─────────────────────────────────────────────── */}
      <section className="py-20">
        <div className="container max-w-3xl px-6">
          <Badge variant="blush" className="mb-6">Origin</Badge>
          <h2 className="text-2xl sm:text-3xl font-semibold text-navy mb-8">Why I built this</h2>

          <div className="space-y-5 text-black/60 leading-relaxed">
            <p>
              I got interested in transplant immunology while reading about how HLA
              mismatch is quantified in clinical practice. The standard approach — counting
              the number of allele mismatches across loci — struck me as a blunt instrument.
              Two alleles can differ by a single amino acid in the peptide-binding groove,
              or they can be from entirely different supertypes. The clinical model treats
              both as "one mismatch."
            </p>
            <p>
              Around the same time I was exploring ESM-2, Meta's protein language model
              trained on 250 million sequences. The embeddings it produces encode
              evolutionary, structural, and functional information that is invisible to
              categorical representations. I wanted to know whether that richer signal
              could improve outcome prediction in transplant patients.
            </p>
            <p>
              CAPA started as a weekend experiment: take the publicly available UCI Bone
              Marrow Transplant dataset, replace the HLA feature columns with ESM-2
              embeddings, and see if a simple survival model trained on top performs
              better than the conventional baseline. The answer was: sometimes — and the
              reasons why are more interesting than the headline number.
            </p>
            <p>
              The architecture grew from there. The current version uses cross-attention
              to explicitly model the <em>interaction</em> between donor and recipient
              allele sets, rather than concatenating their embeddings independently. The
              survival head uses DeepHit to jointly model GvHD, relapse, and
              transplant-related mortality as competing risks — because treating each
              outcome independently with separate Cox models ignores the clinical reality
              that a patient cannot relapse after dying from GvHD.
            </p>
            <p>
              I am publishing this as a proof-of-concept, not a clinical tool. The
              dataset is 187 patients — far too small to train a deep model reliably. The
              conventional Fine–Gray model outperforms DeepHit here, and that result is
              reported honestly in the paper. The value of CAPA at this stage is the
              framework and the question it poses, not the current numbers.
            </p>
          </div>
        </div>
      </section>

      {/* ── Data in depth ────────────────────────────────────────────── */}
      <section className="bg-[#F8F9FC] py-20 border-y border-border">
        <div className="container max-w-7xl px-6">
          <Badge variant="blush" className="mb-6">Dataset</Badge>
          <div className="grid gap-12 lg:grid-cols-[1fr_2fr]">

            <div>
              <h2 className="text-2xl sm:text-3xl font-semibold text-navy mb-4">
                UCI Bone Marrow Transplant Children Dataset
              </h2>
              <p className="text-black/55 leading-relaxed mb-6">
                187 paediatric patients who received allogeneic haematopoietic stem cell
                transplantation. Collected at a single centre in Poland, published by
                Marek Sikora et al. and donated to the UCI Machine Learning Repository.
              </p>

              <div className="space-y-3 mb-6">
                {[
                  ["Patients",      "187 paediatric HSCT recipients"],
                  ["HLA resolution","Antigen level (2-digit) + allele level (4-digit)"],
                  ["Loci covered",  "HLA-A, B, C, DRB1, DQB1"],
                  ["Outcome types", "Survival, acute GvHD, chronic GvHD, relapse, TRM"],
                  ["Follow-up",     "Time-to-event in days from transplant"],
                  ["Missing data",  "Partial — allele-level typings not always available"],
                ].map(([k, v]) => (
                  <div key={k} className="flex gap-4 text-sm">
                    <span className="font-semibold text-navy/60 w-32 shrink-0">{k}</span>
                    <span className="text-foreground">{v}</span>
                  </div>
                ))}
              </div>

              <a
                href="https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 text-sm text-blush-600 hover:text-blush-700 transition-colors"
              >
                UCI Repository
                <ExternalLink className="h-3.5 w-3.5" />
              </a>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-navy/60 uppercase tracking-wider mb-5">
                Feature groups
              </h3>
              <div className="grid gap-4 sm:grid-cols-2">
                {DATASET_FEATURES.map(({ group, items }) => (
                  <div
                    key={group}
                    className="rounded-xl bg-white border border-border/60 p-5 shadow-sm"
                  >
                    <p className="text-xs font-semibold text-blush uppercase tracking-widest mb-3">
                      {group}
                    </p>
                    <ul className="space-y-1.5">
                      {items.map((item) => (
                        <li key={item} className="flex items-start gap-2 text-sm text-black/60">
                          <span className="mt-2 h-1 w-1 rounded-full bg-blush-300 shrink-0" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Technical analysis ───────────────────────────────────────── */}
      <section className="py-20">
        <div className="container max-w-3xl px-6">
          <Badge variant="blush" className="mb-6">Technical Analysis</Badge>
          <h2 className="text-2xl sm:text-3xl font-semibold text-navy mb-8">
            How the model works — and where it struggles
          </h2>

          <div className="space-y-10 text-black/60 leading-relaxed">

            <div>
              <h3 className="text-lg font-semibold text-navy mb-3">ESM-2 as an HLA encoder</h3>
              <p className="mb-3">
                Each HLA allele is mapped to its full protein sequence via IPD-IMGT/HLA,
                then passed through the frozen ESM-2 650M encoder. We use the mean of
                the per-residue representations across the full sequence length as a
                1,280-dimensional allele vector. The encoder is not fine-tuned — the
                pre-trained representations are used as-is and cached to disk in HDF5
                format to avoid re-running inference during training.
              </p>
              <p>
                The key assumption here is that ESM-2's pre-training on diverse protein
                sequences has already learned to encode structural and functional
                similarity in a way that transfers to HLA. This is plausible —
                HLA molecules are peptide-binding proteins, and their binding
                specificity is largely determined by amino acid residues in the
                alpha-1 and alpha-2 groove domains that ESM-2 would have seen in
                homologous sequences. Whether this transfer is truly informative
                for immunological outcome prediction is the central empirical question.
              </p>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-navy mb-3">Cross-attention interaction network</h3>
              <p className="mb-3">
                Rather than concatenating donor and recipient embeddings independently
                (which would lose relational structure), the model uses multi-head
                cross-attention: donor allele embeddings serve as queries, recipient
                embeddings as keys and values. This forces the model to learn which
                donor alleles attend strongly to which recipient alleles — a more
                biologically motivated inductive bias than a simple concatenation.
              </p>
              <p>
                The attention weights are extracted at inference time and visualised
                as a donor × recipient heatmap, giving a rough interpretability window
                into which locus pairs are driving the prediction.
              </p>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-navy mb-3">DeepHit for competing risks</h3>
              <p className="mb-3">
                GvHD, relapse, and transplant-related mortality are competing events:
                the occurrence of one prevents the observation of others. Standard
                survival analysis (Cox-PH per outcome) treats censoring by competing
                events as independent, which is statistically incorrect and leads to
                biased cumulative incidence estimates.
              </p>
              <p>
                DeepHit instead learns a joint discrete-time distribution over all
                event types simultaneously, using a combined log-likelihood loss and a
                ranking loss that penalises incorrect ordering of event times across
                patients. The output is a cumulative incidence function (CIF) per event
                type — the correct estimand for competing-risks data.
              </p>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-navy mb-3">Results and honest interpretation</h3>
              <p className="mb-4">
                Models were evaluated on a held-out test set of 29 patients using
                time-dependent C-index. GvHD C-index was not computable due to
                only 2 events in the test split.
              </p>

              <div className="rounded-xl border border-border overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border bg-muted/40">
                      <th className="text-left px-5 py-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider">Model</th>
                      <th className="text-left px-4 py-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider">C-index Relapse</th>
                      <th className="text-left px-4 py-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider">C-index TRM</th>
                    </tr>
                  </thead>
                  <tbody>
                    {MODEL_RESULTS.map(({ model, relapse, trm, note, highlight }) => (
                      <tr key={model} className={`border-b border-border/60 last:border-0 ${highlight ? "bg-blush-50/50" : ""}`}>
                        <td className="px-5 py-4">
                          <p className={`font-medium ${highlight ? "text-navy" : "text-foreground"}`}>{model}</p>
                          <p className="text-xs text-muted-foreground mt-0.5">{note}</p>
                        </td>
                        <td className="px-4 py-4 font-mono font-semibold text-navy">{relapse}</td>
                        <td className="px-4 py-4 font-mono font-semibold text-navy">{trm}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <p className="mt-4 text-sm text-black/45">
                Fine–Gray is the strongest performing model. DeepHit underperforms
                because 187 patients is insufficient to train a deep network — this is
                expected and the honest result. The framework's value is in the
                architecture and the ESM-2 representation, which would benefit
                substantially from a larger cohort.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── Technical debrief CTA ─────────────────────────────────────── */}
      <section className="bg-[#F8F9FC] py-20 border-y border-border">
        <div className="container max-w-3xl px-6">
          <div className="rounded-2xl border border-border bg-white p-8 shadow-sm">
            <div className="flex items-start gap-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blush-50 border border-blush-200 shrink-0">
                <FileText className="h-5 w-5 text-blush" />
              </div>
              <div className="flex-1">
                <p className="text-xs font-semibold text-blush uppercase tracking-widest mb-1">
                  Technical Debrief · v1
                </p>
                <h3 className="text-xl font-semibold text-navy mb-2">
                  Full writeup — architecture, experiments, and analysis
                </h3>
                <p className="text-black/55 text-sm leading-relaxed mb-5">
                  The technical debrief covers the full model architecture in detail,
                  training procedure, ablation results, embedding space analysis, and
                  discussion of where the approach works and where it doesn't. This is
                  a living document — v1 reflects the current state of the project.
                </p>
                <Button asChild variant="blush">
                  <a
                    href="https://docs.google.com/document/d/1IALiVoo6vYtWGkGcxmqoMyF_gyQNMmClsotXd8b5JxQ/edit?usp=sharing"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Read the Technical Debrief
                    <ExternalLink className="h-3.5 w-3.5" />
                  </a>
                </Button>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Tech stack ────────────────────────────────────────────────── */}
      <section className="py-20">
        <div className="container max-w-7xl px-6">
          <Badge variant="blush" className="mb-6">Stack</Badge>
          <h2 className="text-2xl font-semibold text-navy mb-8">Technical details</h2>
          <div className="grid gap-3 sm:grid-cols-2">
            {TECH.map(({ label, value }) => (
              <div
                key={label}
                className="flex items-start gap-4 rounded-xl bg-[#F8F9FC] border border-border/60 p-4"
              >
                <span className="text-xs font-semibold text-navy/50 uppercase tracking-wider w-28 shrink-0 pt-0.5">
                  {label}
                </span>
                <span className="text-sm text-foreground">{value}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Limitations ───────────────────────────────────────────────── */}
      <section className="bg-[#F8F9FC] py-20 border-t border-border">
        <div className="container max-w-7xl px-6">
          <Badge variant="blush" className="mb-6">Transparency</Badge>
          <h2 className="text-2xl font-semibold text-navy mb-3">Limitations</h2>
          <p className="text-muted-foreground mb-8">
            Honest reporting of what CAPA can and cannot do.
          </p>

          <Card className="border-amber-200 bg-amber-50 max-w-3xl">
            <CardHeader>
              <CardTitle className="text-amber-800 text-base flex items-center gap-2">
                <Microscope className="h-4 w-4" />
                Research prototype — not for clinical use
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-3">
                {LIMITATIONS.map((lim) => (
                  <li key={lim} className="flex items-start gap-2.5 text-sm text-amber-800">
                    <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-amber-500 shrink-0" />
                    {lim}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* ── Contributors ──────────────────────────────────────────────── */}
      <section className="py-20 border-t border-border">
        <div className="container max-w-7xl px-6">
          <Badge variant="blush" className="mb-6">People</Badge>
          <h2 className="text-2xl font-semibold text-navy mb-8">Contributors</h2>
          <div className="flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blush-100 text-blush-600 font-semibold text-lg select-none font-display">
              H
            </div>
            <div>
              <p className="font-semibold text-foreground">Huanxuan Li (Shawn)</p>
              <p className="text-sm text-muted-foreground">Lead developer · Architecture, training pipeline, web interface</p>
            </div>
          </div>
        </div>
      </section>

      {/* ── CTA ───────────────────────────────────────────────────────── */}
      <section className="bg-[#F8F9FC] py-20 border-t border-border">
        <div className="container max-w-7xl px-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6">
          <div>
            <h2 className="text-xl font-semibold text-navy">Interested in contributing?</h2>
            <p className="text-muted-foreground mt-1 text-sm">
              CAPA is open source. Pull requests, issues, and feedback are welcome.
            </p>
          </div>
          <div className="flex gap-3">
            <Button asChild variant="default">
              <a href="https://github.com/sh4wn27/capa" target="_blank" rel="noopener noreferrer">
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
      </section>

    </div>
  );
}
