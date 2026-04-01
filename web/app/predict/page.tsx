import HLAInput from "@/components/HLAInput";
import RiskChart from "@/components/RiskChart";

export default function PredictPage() {
  return (
    <main className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">Transplant Risk Prediction</h1>
      <p className="text-gray-600 mb-8">
        Enter donor and recipient HLA typing to generate competing-risk curves
        for GvHD, relapse, and transplant-related mortality.
      </p>
      <HLAInput />
    </main>
  );
}
