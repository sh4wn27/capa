"use client";

import { useState } from "react";
import { predictRisk, type PredictionRequest, type PredictionResponse } from "@/lib/api";

const LOCI = ["A", "B", "C", "DRB1", "DQB1"] as const;

export default function HLAInput() {
  const [donorHla, setDonorHla] = useState<Record<string, string>>({});
  const [recipientHla, setRecipientHla] = useState<Record<string, string>>({});
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const req: PredictionRequest = {
        donor_hla: donorHla,
        recipient_hla: recipientHla,
      };
      const data = await predictRisk(req);
      setResult(data);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit} className="grid grid-cols-2 gap-8 mb-8">
        <fieldset>
          <legend className="font-semibold text-lg mb-3">Donor HLA</legend>
          {LOCI.map((locus) => (
            <div key={locus} className="mb-2">
              <label className="block text-sm font-medium text-gray-700">{locus}</label>
              <input
                type="text"
                placeholder={`${locus}*02:01`}
                className="mt-1 block w-full rounded border-gray-300 shadow-sm text-sm"
                value={donorHla[locus] ?? ""}
                onChange={(e) =>
                  setDonorHla((prev) => ({ ...prev, [locus]: e.target.value }))
                }
              />
            </div>
          ))}
        </fieldset>

        <fieldset>
          <legend className="font-semibold text-lg mb-3">Recipient HLA</legend>
          {LOCI.map((locus) => (
            <div key={locus} className="mb-2">
              <label className="block text-sm font-medium text-gray-700">{locus}</label>
              <input
                type="text"
                placeholder={`${locus}*02:01`}
                className="mt-1 block w-full rounded border-gray-300 shadow-sm text-sm"
                value={recipientHla[locus] ?? ""}
                onChange={(e) =>
                  setRecipientHla((prev) => ({ ...prev, [locus]: e.target.value }))
                }
              />
            </div>
          ))}
        </fieldset>

        <div className="col-span-2">
          <button
            type="submit"
            disabled={loading}
            className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? "Predicting…" : "Predict Risk"}
          </button>
        </div>
      </form>

      {error && <p className="text-red-600">{error}</p>}
      {result && (
        <div>
          <h2 className="text-xl font-semibold mb-4">Results</h2>
          {/* TODO: render RiskChart and AttentionHeatmap */}
          <pre className="text-xs bg-gray-50 p-4 rounded overflow-auto">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
