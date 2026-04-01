export interface HLATyping {
  A?: string;
  B?: string;
  C?: string;
  DRB1?: string;
  DQB1?: string;
}

export interface ClinicalCovariates {
  age_recipient?: number;
  age_donor?: number;
  disease?: string;
  conditioning?: string;
  donor_type?: string;
}

export interface PredictionRequest {
  donor_hla: HLATyping;
  recipient_hla: HLATyping;
  clinical?: ClinicalCovariates;
}

export interface EventRisk {
  cumulative_incidence: number[];
  risk_score: number;
}

export interface PredictionResponse {
  gvhd: EventRisk;
  relapse: EventRisk;
  trm: EventRisk;
  attention_weights?: number[][] | null;
}

export async function predictRisk(
  request: PredictionRequest
): Promise<PredictionResponse> {
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Prediction failed (${response.status}): ${text}`);
  }

  return response.json() as Promise<PredictionResponse>;
}
