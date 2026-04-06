"use client";

import Link from "next/link";
import { motion, useScroll, useTransform } from "framer-motion";
import { useRef } from "react";
import {
  ArrowRight,
  ChevronRight,
  Dna,
  Brain,
  BarChart2,
  GitMerge,
  BookOpen,
  FlaskConical,
  TrendingUp,
  Users,
  Layers,
} from "lucide-react";
import { GithubIcon } from "@/components/ui/github-icon";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { FadeIn, FadeInStagger, fadeUpItem } from "@/components/FadeIn";

/* ─── shared animation presets ─────────────────────────────────────── */
const ease = [0.21, 0.47, 0.32, 0.98] as [number, number, number, number];

/* ─── data ──────────────────────────────────────────────────────────── */
const STEPS = [
  {
    icon:  Dna,
    n:     "01",
    color: "bg-amber-50 text-amber-600 border-amber-200",
    dot:   "bg-amber-400",
    title: "HLA Input",
    body:  "Donor and recipient HLA alleles at five loci (A, B, C, DRB1, DQB1) are looked up in the IPD-IMGT/HLA database to retrieve their full protein sequences.",
    tag:   "allele → sequence",
  },
  {
    icon:  Brain,
    n:     "02",
    color: "bg-navy/5 text-navy border-navy/15",
    dot:   "bg-navy-500",
    title: "ESM-2 Embedding",
    body:  "Each amino-acid sequence is encoded by frozen ESM-2 (650 M parameters) into a 1 280-dim vector. Structural similarity is preserved — immunologically similar alleles cluster together.",
    tag:   "sequence → 1 280-dim",
  },
  {
    icon:  BarChart2,
    n:     "03",
    color: "bg-blush-50 text-blush-600 border-blush-200",
    dot:   "bg-blush",
    title: "Risk Prediction",
    body:  "A cross-attention network models donor–recipient allele interactions. DeepHit jointly outputs cumulative incidence curves for GvHD, relapse, and TRM as competing events.",
    tag:   "interaction → CIF curves",
  },
];

const RESULTS = [
  {
    metric: "+0.12",
    label:  "C-index over Cox-PH",
    sub:    "GvHD endpoint",
    color:  "text-blush",
    icon:   TrendingUp,
  },
  {
    metric: "+0.07",
    label:  "C-index over RSF",
    sub:    "Relapse endpoint",
    color:  "text-amber-500",
    icon:   TrendingUp,
  },
  {
    metric: "187",
    label:  "Patients (UCI BMT)",
    sub:    "Paediatric HSCT cohort",
    color:  "text-navy",
    icon:   Users,
  },
  {
    metric: "3",
    label:  "Competing risks",
    sub:    "GvHD · Relapse · TRM",
    color:  "text-navy",
    icon:   Layers,
  },
];

const MODEL_ROWS = [
  { name: "Cox-PH",      gvhd: 0.62, relapse: 0.59, trm: 0.60, highlight: false },
  { name: "RSF",         gvhd: 0.66, relapse: 0.62, trm: 0.64, highlight: false },
  { name: "DeepSurv",    gvhd: 0.68, relapse: 0.64, trm: 0.66, highlight: false },
  { name: "CAPA (ours)", gvhd: 0.74, relapse: 0.70, trm: 0.72, highlight: true  },
];

/* ─── sub-components ────────────────────────────────────────────────── */

function CIndexBar({ value, max = 0.8 }: { value: number; max?: number }) {
  const pct = ((value - 0.5) / (max - 0.5)) * 100;
  return (
    <div className="flex items-center gap-2">
      <span className="font-mono text-sm font-semibold w-10 shrink-0">{value.toFixed(2)}</span>
      <div className="flex-1 h-1.5 rounded-full bg-border overflow-hidden">
        <motion.div
          className="h-full rounded-full bg-blush"
          initial={{ width: 0 }}
          whileInView={{ width: `${pct}%` }}
          viewport={{ once: true }}
          transition={{ duration: 0.7, ease, delay: 0.1 }}
        />
      </div>
    </div>
  );
}

/* ─── page ──────────────────────────────────────────────────────────── */
export default function HomePage() {
  const heroRef = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: heroRef, offset: ["start start", "end start"] });
  const heroOpacity = useTransform(scrollYProgress, [0, 0.6], [1, 0]);
  const heroY       = useTransform(scrollYProgress, [0, 0.6], ["0%", "12%"]);

  return (
    <>
      {/* ──────────────────────────────────────────────────────────────
          HERO
      ────────────────────────────────────────────────────────────── */}
      <section ref={heroRef} className="hero-bg relative overflow-hidden min-h-[92vh] flex items-center">

        {/* Radial glow */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              "radial-gradient(ellipse 90% 60% at 50% -10%, rgba(232,132,154,0.18) 0%, transparent 65%)",
          }}
        />

        {/* Parallax content */}
        <motion.div
          style={{ opacity: heroOpacity, y: heroY }}
          className="container relative max-w-7xl px-6 py-28 mx-auto"
        >
          <div className="max-w-4xl mx-auto text-center">

            {/* Status pill */}
            <motion.div
              initial={{ opacity: 0, y: -12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, ease }}
              className="mb-8 flex justify-center"
            >
              <span className="inline-flex items-center gap-2 rounded-full border border-black/10 bg-black/5 px-4 py-1.5 text-xs font-medium text-black/55 backdrop-blur-sm">
                <span className="h-1.5 w-1.5 rounded-full bg-blush-400 animate-pulse-dot" />
                Open Source · MIT · Proof-of-Concept
              </span>
            </motion.div>

            {/* Headline */}
            <motion.h1
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.65, delay: 0.08, ease }}
              className="text-[clamp(2.25rem,6vw,4.5rem)] font-bold leading-[1.08] text-foreground"
            >
              Predicting Alloimmunity with{" "}
              <span className="gradient-text">Protein Language Models</span>
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.18, ease }}
              className="mt-6 text-lg sm:text-xl text-black/55 leading-relaxed max-w-2xl mx-auto"
            >
              CAPA replaces coarse HLA match/mismatch scores with{" "}
              <span className="text-black/80 font-medium">continuous ESM-2 embeddings</span>{" "}
              and predicts GvHD, relapse, and transplant-related mortality as
              competing risks using cross-attention and DeepHit.
            </motion.p>

            {/* CTAs */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.55, delay: 0.28, ease }}
              className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4"
            >
              <Button asChild size="xl" variant="blush" className="w-full sm:w-auto group">
                <Link href="/predict">
                  Open Prediction Tool
                  <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
                </Link>
              </Button>
              <Button
                asChild
                size="xl"
                variant="outline"
                className="w-full sm:w-auto border-black/15 text-foreground hover:bg-black/5"
              >
                <Link href="/paper">
                  <BookOpen className="h-4 w-4" />
                  Read the Paper
                </Link>
              </Button>
            </motion.div>

            {/* GitHub */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.45 }}
              className="mt-8 flex justify-center"
            >
              <a
                href="https://github.com/capa-project/capa"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-sm text-black/35 hover:text-black/65 transition-colors"
              >
                <GithubIcon className="h-4 w-4" />
                capa-project/capa
              </a>
            </motion.div>
          </div>

          {/* Scroll hint */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.2, duration: 0.8 }}
            className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1.5"
          >
            <span className="text-[10px] text-black/25 uppercase tracking-widest">scroll</span>
            <motion.div
              animate={{ y: [0, 5, 0] }}
              transition={{ repeat: Infinity, duration: 1.5, ease: "easeInOut" }}
              className="w-px h-6 bg-gradient-to-b from-black/20 to-transparent"
            />
          </motion.div>
        </motion.div>
      </section>

      {/* ──────────────────────────────────────────────────────────────
          HOW IT WORKS
      ────────────────────────────────────────────────────────────── */}
      <section id="how-it-works" className="bg-white py-28">
        <div className="container max-w-7xl px-6">

          <FadeIn className="text-center mb-18">
            <Badge variant="blush" className="mb-4">How it works</Badge>
            <h2 className="text-3xl sm:text-4xl font-bold text-navy">
              From allele strings to risk curves
            </h2>
            <p className="mt-4 text-muted-foreground max-w-xl mx-auto">
              Three stages transform raw HLA typing into calibrated, interpretable
              competing-risk predictions.
            </p>
          </FadeIn>

          {/* Steps */}
          <div className="mt-16 relative">

            {/* Connector line — desktop only */}
            <div className="hidden lg:block absolute top-[3.25rem] left-[calc(16.66%-1.5rem)] right-[calc(16.66%-1.5rem)] h-px">
              <motion.div
                className="h-full bg-gradient-to-r from-amber-300 via-navy/40 to-blush-300"
                initial={{ scaleX: 0, originX: 0 }}
                whileInView={{ scaleX: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 1.1, ease, delay: 0.3 }}
              />
            </div>

            <FadeInStagger className="grid gap-8 lg:grid-cols-3" stagger={0.12} delay={0.1}>
              {STEPS.map(({ icon: Icon, n, color, dot, title, body, tag }) => (
                <motion.div key={n} variants={fadeUpItem}>
                  <Card className="relative overflow-hidden border-border/60 shadow-sm hover:shadow-md transition-shadow group">
                    <CardContent className="pt-7 pb-7 px-7">
                      {/* Step number watermark */}
                      <span className="pointer-events-none select-none absolute -top-2 -right-1 text-[6rem] font-black text-navy/4 leading-none">
                        {n}
                      </span>

                      {/* Icon */}
                      <div className={`flex h-12 w-12 items-center justify-center rounded-xl border ${color} mb-5 shadow-sm group-hover:scale-105 transition-transform`}>
                        <Icon className="h-5 w-5" />
                      </div>

                      {/* Badge */}
                      <span className="inline-flex items-center gap-1.5 rounded-full bg-muted px-2.5 py-0.5 text-[10px] font-mono text-muted-foreground border border-border mb-3">
                        <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
                        {tag}
                      </span>

                      <h3 className="text-lg font-semibold text-navy mb-2">{title}</h3>
                      <p className="text-sm text-muted-foreground leading-relaxed">{body}</p>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </FadeInStagger>
          </div>

          {/* Architecture mini-diagram */}
          <FadeIn delay={0.2} className="mt-16">
            <div className="rounded-2xl border border-border bg-[#F8F9FC] p-6 sm:p-8 font-mono text-xs overflow-x-auto">
              <div className="flex items-stretch gap-0 min-w-[520px]">

                {/* Input box */}
                <div className="flex-1 rounded-xl border border-navy/12 bg-white p-4 shadow-sm">
                  <p className="text-[9px] uppercase tracking-widest text-muted-foreground mb-2">Input</p>
                  <p className="font-sans font-semibold text-navy text-sm leading-snug">
                    HLA-A*02:01<br />HLA-B*07:02<br />HLA-DRB1*15:01
                  </p>
                </div>

                {/* Arrow */}
                <div className="flex items-center px-3 text-muted-foreground/30">
                  <ChevronRight className="h-5 w-5" />
                </div>

                {/* ESM box */}
                <div className="flex-1 rounded-xl border border-amber-200 bg-amber-50 p-4 shadow-sm">
                  <p className="text-[9px] uppercase tracking-widest text-amber-600 mb-2">ESM-2 · 650M</p>
                  <p className="font-sans font-semibold text-navy text-sm leading-snug">
                    1 280-dim<br />embedding<br />per allele
                  </p>
                </div>

                <div className="flex items-center px-3 text-muted-foreground/30">
                  <ChevronRight className="h-5 w-5" />
                </div>

                {/* Cross-attention box */}
                <div className="flex-1 rounded-xl border border-navy/15 bg-navy/5 p-4 shadow-sm">
                  <p className="text-[9px] uppercase tracking-widest text-navy/50 mb-2">Cross-Attention</p>
                  <p className="font-sans font-semibold text-navy text-sm leading-snug">
                    Donor × Recipient<br />interaction<br />128-dim
                  </p>
                </div>

                <div className="flex items-center px-3 text-muted-foreground/30">
                  <ChevronRight className="h-5 w-5" />
                </div>

                {/* Output box */}
                <div className="flex-1 rounded-xl border border-blush/25 bg-blush-50 p-4 shadow-sm">
                  <p className="text-[9px] uppercase tracking-widest text-blush-600 mb-2">DeepHit output</p>
                  <div className="flex flex-col gap-0.5 mt-1">
                    {["GvHD CIF", "Relapse CIF", "TRM CIF"].map((ev, i) => (
                      <div key={ev} className="flex items-center gap-1.5">
                        <span className={`h-1.5 w-1.5 rounded-full shrink-0 ${["bg-orange-400","bg-blue-500","bg-red-400"][i]}`} />
                        <span className="font-sans text-xs text-navy/80">{ev}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </FadeIn>
        </div>
      </section>

      {/* ──────────────────────────────────────────────────────────────
          KEY RESULTS
      ────────────────────────────────────────────────────────────── */}
      <section id="results" className="bg-[#F8F9FC] py-28 border-y border-border">
        <div className="container max-w-7xl px-6">

          <FadeIn className="text-center mb-16">
            <Badge variant="blush" className="mb-4">Key results</Badge>
            <h2 className="text-3xl sm:text-4xl font-bold text-navy">
              Outperforming traditional baselines
            </h2>
            <p className="mt-4 text-muted-foreground max-w-xl mx-auto">
              Evaluated on the UCI Bone Marrow Transplant dataset (n = 187) using
              time-dependent C-index and Brier score.
            </p>
          </FadeIn>

          {/* Metric cards */}
          <FadeInStagger className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4 mb-16" stagger={0.1}>
            {RESULTS.map(({ metric, label, sub, color, icon: Icon }) => (
              <motion.div key={label} variants={fadeUpItem}>
                <Card className="border-border/60 shadow-sm hover:shadow-md transition-shadow text-center py-7 px-6">
                  <CardContent className="p-0">
                    <div className="flex justify-center mb-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-muted">
                        <Icon className={`h-5 w-5 ${color}`} />
                      </div>
                    </div>
                    <p className={`text-4xl font-black ${color}`}>{metric}</p>
                    <p className="text-sm font-semibold text-navy mt-1">{label}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">{sub}</p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </FadeInStagger>

          {/* C-index comparison table */}
          <FadeIn delay={0.1}>
            <div className="rounded-2xl border border-border bg-white shadow-sm overflow-hidden">
              <div className="px-6 py-4 border-b border-border flex items-center justify-between">
                <h3 className="font-semibold text-navy text-sm">
                  Time-dependent C-index comparison
                </h3>
                <span className="text-xs text-muted-foreground">UCI BMT · n = 187</span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border bg-muted/40">
                      <th className="text-left px-6 py-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                        Model
                      </th>
                      {["GvHD", "Relapse", "TRM"].map((ev) => (
                        <th key={ev} className="text-left px-4 py-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                          {ev}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {MODEL_ROWS.map(({ name, gvhd, relapse, trm, highlight }) => (
                      <tr
                        key={name}
                        className={`border-b border-border/60 last:border-0 ${
                          highlight
                            ? "bg-blush-50/60"
                            : "hover:bg-muted/30 transition-colors"
                        }`}
                      >
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-2">
                            <span className={`font-medium ${highlight ? "text-navy" : "text-foreground"}`}>
                              {name}
                            </span>
                            {highlight && (
                              <Badge variant="blush" className="text-[9px] py-0">
                                best
                              </Badge>
                            )}
                          </div>
                        </td>
                        {[gvhd, relapse, trm].map((v, i) => (
                          <td key={i} className="px-4 py-4 w-40">
                            <CIndexBar value={v} />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </FadeIn>
        </div>
      </section>

      {/* ──────────────────────────────────────────────────────────────
          ABOUT
      ────────────────────────────────────────────────────────────── */}
      <section id="about" className="bg-white py-28">
        <div className="container max-w-7xl px-6">
          <div className="grid gap-16 lg:grid-cols-2 items-center">

            {/* Text */}
            <FadeIn direction="left">
              <Badge variant="blush" className="mb-5">About the project</Badge>
              <h2 className="text-3xl sm:text-4xl font-bold text-navy leading-snug">
                A new lens on HLA compatibility
              </h2>
              <p className="mt-5 text-muted-foreground leading-relaxed">
                Haematopoietic stem cell transplantation outcome depends critically on
                HLA compatibility. The standard approach encodes this as a binary
                match/mismatch count, discarding most immunological information.
              </p>
              <p className="mt-4 text-muted-foreground leading-relaxed">
                CAPA was built to change that. By encoding every allele with ESM-2 —
                a protein language model trained on 250 M sequences — and learning
                donor–recipient interaction through cross-attention, we get embeddings
                that reflect structural and functional similarity rather than mere
                categorical identity.
              </p>
              <p className="mt-4 text-muted-foreground leading-relaxed">
                This is an open-source proof-of-concept, validated on 187 paediatric
                HSCT patients. We acknowledge the small cohort limitation and encourage
                replication on larger datasets.
              </p>
              <div className="mt-8 flex flex-wrap gap-3">
                <Button asChild variant="default">
                  <Link href="/about">
                    Full project story
                    <ArrowRight className="h-4 w-4" />
                  </Link>
                </Button>
                <Button asChild variant="outline">
                  <Link href="/paper">
                    <BookOpen className="h-4 w-4" />
                    Read the paper
                  </Link>
                </Button>
              </div>
            </FadeIn>

            {/* Stats / features grid */}
            <FadeIn direction="right" delay={0.1}>
              <div className="grid grid-cols-2 gap-4">
                {[
                  { icon: Dna,       label: "ESM-2 Embeddings",  desc: "1 280-dim per allele, frozen 650M model" },
                  { icon: Brain,     label: "Cross-Attention",    desc: "Interpretable donor × recipient interaction" },
                  { icon: BarChart2, label: "DeepHit Survival",   desc: "Joint competing-risks CIF output" },
                  { icon: GitMerge,  label: "Open Source",        desc: "MIT licensed, fully reproducible" },
                ].map(({ icon: Icon, label, desc }, i) => (
                  <motion.div
                    key={label}
                    whileHover={{ y: -2, boxShadow: "0 8px 24px rgba(15,28,53,0.08)" }}
                    className="rounded-xl border border-border/60 bg-[#F8F9FC] p-5 transition-all cursor-default"
                  >
                    <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-white border border-border shadow-sm mb-3">
                      <Icon className="h-4 w-4 text-navy/60" />
                    </div>
                    <p className="font-semibold text-navy text-sm">{label}</p>
                    <p className="text-xs text-muted-foreground mt-1 leading-relaxed">{desc}</p>
                  </motion.div>
                ))}
              </div>
            </FadeIn>
          </div>
        </div>
      </section>

      {/* ──────────────────────────────────────────────────────────────
          CTA BANNER
      ────────────────────────────────────────────────────────────── */}
      <section className="relative overflow-hidden py-24 bg-[#111111]">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              "radial-gradient(ellipse 65% 80% at 50% 110%, rgba(232,132,154,0.12) 0%, transparent 60%)",
          }}
        />
        <FadeIn className="container relative max-w-2xl px-6 text-center">
          <FlaskConical className="mx-auto mb-5 h-9 w-9 text-blush opacity-70" />
          <h2 className="text-3xl sm:text-4xl font-bold text-white">
            Try it on your own data
          </h2>
          <p className="mt-4 text-white/52 leading-relaxed">
            Enter donor and recipient HLA strings and receive competing-risk curves,
            attention heatmaps, and SHAP feature attribution in seconds.
          </p>
          <div className="mt-10 flex flex-col sm:flex-row justify-center gap-4">
            <Button asChild size="xl" variant="blush" className="group">
              <Link href="/predict">
                Open Prediction Tool
                <ChevronRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
              </Link>
            </Button>
            <Button
              asChild
              size="xl"
              variant="outline"
              className="border-white/18 bg-white/5 text-white hover:bg-white/10 hover:text-white"
            >
              <a href="https://github.com/capa-project/capa" target="_blank" rel="noopener noreferrer">
                <GithubIcon className="h-4 w-4" />
                View source
              </a>
            </Button>
          </div>
        </FadeIn>
      </section>

      {/* ──────────────────────────────────────────────────────────────
          FOOTER
      ────────────────────────────────────────────────────────────── */}
      <footer className="bg-navy border-t border-white/8">
        <div className="container max-w-7xl px-6 py-14">
          <div className="grid gap-10 sm:grid-cols-2 lg:grid-cols-4">

            {/* Brand */}
            <div className="lg:col-span-2">
              <div className="flex items-center gap-2.5 mb-4">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-white/10 text-white font-bold text-sm">
                  C
                </div>
                <span className="font-semibold text-white text-lg">CAPA</span>
              </div>
              <p className="text-sm text-white/40 leading-relaxed max-w-xs">
                Computational Architecture for Predicting Alloimmunity. An
                open-source proof-of-concept for structure-aware HLA mismatch
                modelling.
              </p>
              <div className="mt-5 flex items-center gap-4">
                <a
                  href="https://github.com/capa-project/capa"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-white/35 hover:text-white/70 transition-colors"
                  aria-label="GitHub"
                >
                  <GithubIcon className="h-5 w-5" />
                </a>
              </div>
            </div>

            {/* Links */}
            <div>
              <p className="text-xs font-semibold text-white/30 uppercase tracking-wider mb-4">
                Navigation
              </p>
              <ul className="space-y-2.5">
                {[
                  { href: "/",        label: "Home"    },
                  { href: "/predict", label: "Predict" },
                  { href: "/about",   label: "About"   },
                  { href: "/paper",   label: "Paper"   },
                ].map(({ href, label }) => (
                  <li key={href}>
                    <Link
                      href={href}
                      className="text-sm text-white/45 hover:text-white/80 transition-colors"
                    >
                      {label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>

            {/* Resources */}
            <div>
              <p className="text-xs font-semibold text-white/30 uppercase tracking-wider mb-4">
                Resources
              </p>
              <ul className="space-y-2.5">
                {[
                  {
                    href:  "https://github.com/capa-project/capa",
                    label: "GitHub Repository",
                  },
                  {
                    href:  "https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children",
                    label: "UCI BMT Dataset",
                  },
                  {
                    href:  "https://www.ebi.ac.uk/ipd/imgt/hla/",
                    label: "IPD-IMGT/HLA",
                  },
                  {
                    href:  "https://huggingface.co/facebook/esm2_t33_650M_UR50D",
                    label: "ESM-2 Model",
                  },
                ].map(({ href, label }) => (
                  <li key={href}>
                    <a
                      href={href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-white/45 hover:text-white/80 transition-colors"
                    >
                      {label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Bottom bar */}
          <div className="mt-12 pt-6 border-t border-white/8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-white/25">
            <p>© {new Date().getFullYear()} Huanxuan Li (Shawn) · MIT License</p>
            <p>
              Built with Next.js · Tailwind CSS · ESM-2 · PyTorch
            </p>
          </div>
        </div>
      </footer>
    </>
  );
}
