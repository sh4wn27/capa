export default function AboutPage() {
  return (
    <main className="container mx-auto px-4 py-8 max-w-3xl">
      <h1 className="text-3xl font-bold mb-4">About CAPA</h1>
      <p className="text-gray-700 leading-relaxed mb-4">
        CAPA (Computational Architecture for Predicting Alloimmunity) replaces
        categorical HLA match/mismatch scoring with continuous, biologically
        meaningful embeddings derived from the ESM-2 protein language model.
      </p>
      <p className="text-gray-700 leading-relaxed mb-4">
        The model uses cross-attention to capture the immunological &ldquo;conflict&rdquo;
        between donor and recipient HLA allele pairs, and DeepHit to jointly
        model GvHD, relapse, and transplant-related mortality as competing risks.
      </p>
      {/* TODO: Add team section and story */}
    </main>
  );
}
