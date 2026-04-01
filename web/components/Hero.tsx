import Link from "next/link";

export default function Hero() {
  return (
    <section className="flex flex-col items-center justify-center min-h-screen px-4 text-center bg-gradient-to-b from-white to-blue-50">
      <h1 className="text-5xl font-bold text-gray-900 mb-4">
        CAPA
      </h1>
      <p className="text-xl text-gray-600 max-w-2xl mb-2">
        Computational Architecture for Predicting Alloimmunity
      </p>
      <p className="text-gray-500 max-w-xl mb-8">
        Structure-aware HLA mismatch representations for post-transplant outcome
        prediction using protein language models and deep competing-risks survival analysis.
      </p>
      <div className="flex gap-4">
        <Link
          href="/predict"
          className="px-8 py-3 bg-blue-600 text-white rounded-lg text-lg font-medium hover:bg-blue-700 transition"
        >
          Try the Tool
        </Link>
        <Link
          href="/about"
          className="px-8 py-3 border border-gray-300 text-gray-700 rounded-lg text-lg font-medium hover:bg-gray-50 transition"
        >
          Learn More
        </Link>
      </div>
    </section>
  );
}
