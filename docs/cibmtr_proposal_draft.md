# CIBMTR Working Committee Study Proposal — DRAFT

> **Status:** draft for mentor review. Not committed to the public repo.
> **Route:** CIBMTR Study Proposal Form → Immunobiology Working Committee (IBWC).
> **Cycle:** the 2027 Tandem deadline (21 Sep 2026) has passed. Target the 2028 cycle (deadline TBD; watch the "Propose a Working Committee Study" page). Send the pre-submission email below now.
> Section headings follow the official CIBMTR *Working Committee Study Proposal Outline*, in the same order.
> `[[ ... ]]` = must be filled in before submission.

---

## 1. Study Title

Directional, sequence-based representation of donor–recipient HLA mismatch and acute GvHD after unrelated-donor hematopoietic cell transplantation

## 2. Key Words

HLA mismatch; HLA-DPB1; T-cell epitope; graft-versus-host direction; protein language model; acute GvHD; unrelated donor

## 3. Principal Investigator Information

| | PI 1 | PI 2 (senior / mentor) |
|---|---|---|
| Name, degrees | Shawn Li | `[[mentor name, MD/PhD]]` |
| Email | `[[contact email]]` | `[[mentor email]]` |
| Institution | `[[school / affiliated lab]]` | `[[mentor institution]]` |
| Academic rank | Student investigator | `[[rank]]` |
| Junior investigator | Yes | `[[yes/no]]` |
| Current CIBMTR work | Analyses of public datasets IB20-02, IB19-01a, IB19-01b (not CIBMTR-sponsored studies) | `[[if any]]` |

> **Note to self / mentor:** a senior co-PI with an academic transplant or HLA appointment is
> effectively required for feasibility review, IRB/data-use agreement, and presenting at Tandem.
> Line this person up before contacting IBWC.

## 4. Proposed Working Committee

Immunobiology. (Alternative: Graft-versus-Host Disease. IBWC is preferred because the exposure is HLA and the study needs donor–recipient genotypes.)

## 5. Research Question

In unrelated-donor HCT, does a **directional**, sequence-based representation of donor–recipient HLA mismatch (graft-versus-host vs. host-versus-graft) predict acute GvHD and transplant-related mortality beyond allele-level mismatch counts and established mismatch-permissiveness models (DPB1 T-cell-epitope groups, eplet load)?

## 6. Research Hypothesis

1. **H1 (direction):** among HLA-mismatched pairs, the GvH-direction component of mismatch is associated with grade II–IV aGvHD independently of the HvG component and of total (symmetric) mismatch.
2. **H2 (structure within category):** within a single mismatch category (e.g., TCE-permissive DPB1, or single-allele 7/8 mismatch at one locus), continuous protein-sequence similarity between the mismatched alleles stratifies TRM and aGvHD risk.
3. **H3 (replication):** two signals seen in public CIBMTR data replicate in an independent, non-overlapping cohort:
   - core-TCE3 DPB1 mismatches carry lower TRM than other TCE3-permissive mismatches;
   - GvH-non-permissive DPB1 mismatches carry more aGvHD II–IV than HvG-non-permissive ones.

## 7. Short Summary of Project

Donor selection treats HLA mismatches as categories, such as "8/8" or "permissive", and as symmetric between donor and recipient. Graft-versus-host alloreactivity is directional, though: donor T cells respond to recipient antigens the donor lacks. In public CIBMTR datasets we found three things.

- **Direction is pervasive.** GvH and HvG eplet loads differ in 82–96% of haploidentical pairs.
- **Direction adds only a borderline signal for aGvHD.** In 5,108 unrelated-donor transplants, GvH- vs HvG-non-permissive DPB1 mismatch gave aGvHD II–IV HR 1.13 (1.00–1.28).
- **Residue-level structure separates risk within the TCE "permissive" category.** Core TCE3 mismatches had lower TRM (HR 0.74, 0.61–0.90).

Public files contain only coarse summaries and no allele strings, so they cannot test a full sequence-based representation. We request 2-field donor and recipient HLA genotypes with standard outcomes for a recent, non-overlapping cohort. With these data we will:

1. replicate the DPB1 findings independently;
2. test directional, protein-language-model-derived mismatch features against standard Cox models and established permissiveness scores.

## 8. Specific Objectives / Outcomes to be Investigated

**Primary outcome:** grade II–IV acute GvHD (cumulative incidence; death without aGvHD as a competing event).
**Secondary outcomes:** grade III–IV aGvHD, chronic GvHD, relapse, TRM, overall survival, neutrophil engraftment.

- **Aim 1. Independent replication of DPB1 findings.**
  - Groups: TCE match, permissive (split core-TCE3 / other), GvH-non-permissive, HvG-non-permissive.
  - Pre-specified contrasts: (a) GvH- vs HvG-non-permissive for aGvHD II–IV; (b) core-TCE3 vs other TCE3-permissive for TRM.
- **Aim 2. Directional mismatch at classical loci.**
  - Population: single-allele-mismatched pairs (7/8) at HLA-A, -B, -C or -DRB1.
  - Test whether the GvH-direction component improves on the symmetric representation, given the mismatched locus. Two directional measures:
    - directional eplet load, computed with HLAMatchmaker/PELC from 2-field typing;
    - the signed difference of ESM-2 protein-language-model embeddings.
- **Aim 3. Learned representation vs. standard methods.**
  - Benchmark CAPA (frozen ESM-2 allele embeddings → signed-difference locus attention → competing-risks survival head) against cause-specific Cox and Fine–Gray models.
  - Comparators use the same covariates plus (i) allele mismatch counts, (ii) TCE group and eplet load.

## 9. Scientific Impact

- **Donor selection.** If the GvH component, or structure within a mismatch category, carries independent risk, it can rank otherwise equivalent mismatched unrelated donors. This matters most for patients without an 8/8 donor, who are disproportionately from under-represented ancestries.
- **Robustness.** A null result in a well-powered independent cohort would also be informative. It would show that current symmetric, categorical scoring loses little, and would limit the clinical claims that sequence-based HLA models can make.

## 10. Scientific Justification

**What is established.**
- HLA-DPB1 TCE-group matching modifies aGvHD and TRM risk after unrelated-donor HCT (Fleischhauer et al. 2012).
- HLA-DP expression level modifies GvHD risk (Petersdorf et al. 2015).

**Current limitations.**
- Permissiveness models are hand-built categories.
- Eplet counts are unweighted tallies.
- Most are scored as symmetric totals, although alloreactivity has a direction.

**Prior work by the applicants (public CIBMTR data, manuscript in preparation: https://github.com/sh4wn27/capa):**
- **IB20-02** (n=1,287 haploidentical, PTCy):
  - GvH and HvG eplet loads were nearly uncorrelated (r = −0.08 to 0.03).
  - Direction did not add to total load for any outcome (LRT p = 0.18–0.72), consistent with Zou et al. 2022.
  - PTCy is the setting where HLA effects are expected to be weakest.
- **IB19-01a** (n=5,108 unrelated-donor):
  - TCE matching replicated (aGvHD II–IV LRT p = 1.5×10⁻⁵).
  - GvH- vs HvG-non-permissive aGvHD II–IV HR was 1.13 (1.00–1.28), p = 0.07.
- **IB19-01a + IB19-01b** (same patients):
  - Within 2,205 TCE3-permissive mismatches, core-TCE3 mismatches had lower TRM (HR 0.74, 0.61–0.90) and mortality (0.79, 0.68–0.92).
  - For aGvHD II–IV, core mismatches resembled DPB1 match (HR 1.11) while other TCE3-permissive mismatches resembled non-permissive (1.27–1.29).
  - This is a re-analysis of the cohort in which the core group was described, so it needs independent replication.
- **Embedding validity:** ESM-2 embedding distances between alleles correlate with EpRegistry eplet loads (Spearman ρ = 0.53–0.66 across five loci; 8,810 allele pairs). 9–25% of directional eplet-load variance cannot be captured by any symmetric pair function.

**Why new data are needed.** All public datasets provide only derived categories, and none provides allele strings. Testing a sequence-based representation, and replicating the DPB1 signals outside 2008–2017, requires donor and recipient genotypes.

## 11. Participant Selection Criteria

**Inclusion**
- First allogeneic HCT, `[[2018]]`–`[[most recent complete year]]`. This does not overlap IB19-01 (2008–2017).
- AML, ALL, or MDS.
- Adult or pediatric recipients.
- Unrelated adult donor (bone marrow or PBSC).
- High-resolution (≥2-field) donor and recipient typing at HLA-A, -B, -C, -DRB1, -DQB1 and -DPB1.
- CRF-level reporting, so that GvHD grade and cause of death are available.

**Exclusion**
- Cord blood.
- Ex vivo T-cell depletion or CD34 selection.
- Prior allogeneic HCT.
- Missing high-resolution typing at any of the six loci.

**GvHD prophylaxis:** analysed in strata, calcineurin-inhibitor based vs PTCy-based, with PTCy pre-specified as a separate stratum. PTCy is now common in unrelated-donor HCT and is expected to blunt HLA effects.

**Feasibility and power (for the statistician to confirm):**
- To detect the IB19-01a directional DPB1 contrast (HR 1.13) at 80% power, two-sided α = 0.05, the Schoenfeld approximation needs ≈2,100 aGvHD II–IV events among non-permissive pairs.
- At the IB19-01a rates (37% non-permissive; 48% aGvHD II–IV), that is ≈11,800 eligible transplants.
- If the cohort is smaller, Aim 1a becomes an estimation aim, reporting the confidence interval rather than a powered test.
- The other two pre-specified contrasts, core-TCE3 vs other TCE3 for TRM (HR 0.74) and Aim 2, need far fewer events.

## 12. Data Requirements (standard CRF variables)

**Patient**
- Age.
- Sex.
- Karnofsky/Lansky score.
- HCT-CI.
- Race/ethnicity.
- CMV serostatus (recipient).

**Disease**
- Disease.
- Disease status at HCT.
- Disease risk index.
- Time from diagnosis to HCT.
- Cytogenetic risk (where available).

**Donor / graft**
- Donor age.
- Donor sex and donor–recipient sex match.
- Donor parity.
- Donor CMV serostatus.
- Graft source.
- Donor–recipient race match.

**Transplant**
- Year.
- Conditioning intensity.
- TBI use.
- GvHD prophylaxis.
- In vivo T-cell depletion (ATG/alemtuzumab).
- Center (pseudo-ID, for clustering).

**Outcomes (each with event time)**
- aGvHD grade II–IV and III–IV.
- Chronic GvHD.
- Relapse.
- TRM.
- Death and cause of death.
- Neutrophil engraftment.

## 13. Additional Variables Needed

1. **2-field donor and recipient HLA genotypes**, both alleles, at HLA-A, -B, -C, -DRB1, -DQB1 and -DPB1.
   - This is the critical request. Allele-level genotypes are needed to compute the sequence-based and eplet features.
   - If release of full genotypes is restricted for de-identification, we propose an alternative: CIBMTR runs the open-source feature pipeline (https://github.com/sh4wn27/capa) internally and releases only derived per-patient features.
2. **HLA-DPB1 TCE group and the expression-associated rs9277534 marker**, if already in the CIBMTR database.
3. **HLA-DPA1 typing**, if available. It is used only in sensitivity analyses.

## 14. Patient-Reported Outcome (PRO) Requirements

Not applicable.

## 15. Machine Learning

**Yes.**

**Model**
- Each HLA allele's protein sequence (IPD-IMGT/HLA release 3.57) is embedded with the frozen ESM-2 650M protein language model (Lin et al. 2023).
- Per locus, the signed donor–recipient embedding difference is computed. It is direction-preserving: it is non-zero for GvH-only and HvG-only mismatches and distinguishes the two.
- A compact attention network (≈470K parameters) operates over loci.
- A DeepHit competing-risks head (Lee et al. 2018) estimates cumulative incidence of aGvHD, relapse and TRM.

**Safeguards for the cohort size**
1. The primary hypothesis tests (H1–H3) use **pre-specified cause-specific Cox models** with low-dimensional, fixed features: directional eplet load and the principal components of the embedding difference. They do not depend on the neural network.
2. The neural model is evaluated only by repeated nested cross-validation, with center-grouped folds.
3. As required, every ML result is benchmarked against cause-specific Cox and Fine–Gray models on identical covariates.

**Metrics**
- Harrell's and time-dependent C-index.
- Integrated Brier score.
- Calibration.

All code is open source and will be shared with the CIBMTR statistician.

## 16. Sample Requirements

Not applicable. No biospecimens are requested.

## 17. Non-CIBMTR Data Source

IPD-IMGT/HLA allele sequences (public reference database) are used only to map allele names to protein sequences. No patient-level external data are linked.

## 18. References

1. Fleischhauer K, et al. Effect of T-cell-epitope matching at HLA-DPB1 in recipients of unrelated-donor haemopoietic-cell transplantation: a retrospective study. *Lancet Oncol.* 2012;13(4):366–374. doi:10.1016/S1470-2045(12)70004-3
2. Petersdorf EW, et al. High HLA-DP expression and graft-versus-host disease. *N Engl J Med.* 2015;373(7):599–609. doi:10.1056/NEJMoa1500140
3. Zou J, et al. Number of HLA-mismatched eplets is not associated with major outcomes in haploidentical transplantation with post-transplantation cyclophosphamide. *Transplant Cell Ther.* 2022;28:107.e1–107.e8. doi:10.1016/j.jtct.2021.11.001
4. Duquesnoy RJ. HLAMatchmaker: a molecularly based algorithm for histocompatibility determination. I. Description of the algorithm. *Hum Immunol.* 2002.
5. Lin Z, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science.* 2023.
6. Lee C, et al. DeepHit: a deep learning approach to survival analysis with competing risks. *Proc AAAI Conf Artif Intell.* 2018.
7. Lhotte R, et al. Python Eplet Load Calculator (PELC). 2022. doi:10.5281/zenodo.7526198
8. Mayor NP, et al. Impact of previously unrecognized HLA mismatches using ultrahigh resolution typing in unrelated donor hematopoietic cell transplantation. *J Clin Oncol.* 2021;39(21):2397–2409. doi:10.1200/JCO.20.03643 (source of IB19-01a, P-5390)
9. Arrieta-Bolaños E, et al. A core group of structurally similar HLA-DPB1 alleles drives permissiveness after hematopoietic cell transplantation. *Blood.* 2022;140(6):659–663. doi:10.1182/blood.2022015708 (source of IB19-01b, P-5597; defined the core TCE3 group a priori by sequence clustering + in vitro alloreactivity, then tested it on this same 5,140-patient cohort)

## 19. Conflicts of Interest

None. `[[mentor to confirm their own]]`

---

## Pre-submission email (send now)

**To:** immunobiology@nmdp.org
`[[cc: IBWC co-chairs / scientific director — names from the current IBWC page]]`

**Subject:** Pre-submission inquiry: directional HLA mismatch representation and aGvHD (Immunobiology WC)

> Dear Immunobiology Working Committee leadership,
>
> I am preparing a Working Committee proposal for the 2028 cycle, with `[[mentor name, institution]]` as senior investigator, and would value your guidance on feasibility and overlap before submission.
>
> Using CIBMTR public datasets IB20-02, IB19-01a and IB19-01b, we found three things:
> - GvH- and HvG-direction mismatch are nearly uncorrelated in real typing.
> - Directional DPB1 non-permissiveness shows a borderline aGvHD II–IV signal (HR 1.13, 1.00–1.28).
> - Core-TCE3 DPB1 mismatches carry lower TRM than other permissive mismatches (HR 0.74, 0.61–0.90).
>
> These public files contain no allele strings, so we cannot test a sequence-based (protein-language-model) mismatch representation or replicate these signals independently.
>
> We would request 2-field donor/recipient genotypes at six loci with standard outcomes for unrelated-donor HCT from `[[2018]]` onward, non-overlapping with IB19-01. If genotype release is restricted, we could provide our open-source feature pipeline for internal computation.
>
> Could you advise:
> 1. whether this overlaps with active IBWC studies;
> 2. the approximate number of eligible transplants with six-locus high-resolution typing;
> 3. whether derived-feature release is an acceptable alternative to genotype release?
>
> A one-page summary is attached. Thank you for your time.
>
> Shawn Li
> `[[affiliation, contact]]`

---

## Checklist before submitting

- [ ] Senior co-PI agreed and named (Section 3)
- [x] IB19-01a/b source publications identified and cited (refs 8–9). Core TCE3 was defined a priori (sequence clustering), but first tested clinically on this same cohort, so Aim 1b is a true replication only on a non-overlapping cohort
- [ ] Year range set after IBWC reply on available N (Section 11)
- [ ] Statistician-facing power numbers re-checked against the cohort table CIBMTR provides
- [ ] Review CIBMTR Study Proposal Guidance, PI checklist, TED vs CRF reporting resource, MCW/NMDP COI policies
- [ ] IRB / data-use agreement route confirmed with mentor's institution
