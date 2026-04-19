export interface HLATyping {
  A?: string;
  B?: string;
  C?: string;
  DRB1?: string;
  DQB1?: string;
  DPB1?: string;
}

export interface ClinicalCovariates {
  age_recipient?:   number;
  age_donor?:       number;
  cd34_dose?:       number;  // ×10⁶/kg
  sex_mismatch?:    boolean;
  disease?:         string;
  conditioning?:    string;
  donor_type?:      string;
  stem_cell_source?: string;
}

export interface PredictionRequest {
  donor_hla:     HLATyping;
  recipient_hla: HLATyping;
  clinical?:     ClinicalCovariates;
}

export interface EventRisk {
  cumulative_incidence: number[];  // length = time bins (0…730 days)
  risk_score: number;              // scalar in [0,1]
  time_points?: number[];          // optional day axis
}

export interface PredictionResponse {
  gvhd:    EventRisk;
  relapse: EventRisk;
  trm:     EventRisk;
  attention_weights?: number[][] | null;  // (n_loci_donor, n_loci_recip)
  mismatch_count?: number;
  model_version?: string;
}

// ── Multi-donor comparison ────────────────────────────────────────────────────

export interface DonorEntry {
  label?: string;
  donor_hla: HLATyping;
}

export interface ComparisonRequest {
  recipient_hla: HLATyping;
  donors: DonorEntry[];
  clinical?: ClinicalCovariates;
}

export interface DonorRiskSummary {
  label: string;
  gvhd_risk: number;
  relapse_risk: number;
  trm_risk: number;
  mismatch_count?: number;
  rank: number;
  full_prediction: PredictionResponse;
}

export interface ComparisonResponse {
  donors: DonorRiskSummary[];
  best_donor_label: string;
  model_version?: string;
}

export async function comparedonors(
  request: ComparisonRequest
): Promise<ComparisonResponse> {
  const response = await fetch("/api/compare", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(request),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Comparison failed (${response.status}): ${text}`);
  }

  return response.json() as Promise<ComparisonResponse>;
}

// ── Single-patient prediction ─────────────────────────────────────────────────

export async function predictRisk(
  request: PredictionRequest
): Promise<PredictionResponse> {
  const response = await fetch("/api/predict", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(request),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Prediction failed (${response.status}): ${text}`);
  }

  return response.json() as Promise<PredictionResponse>;
}
