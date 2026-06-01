/* ============================================================
   CAPA — synthetic inference engine (deterministic, demo only)
   Produces plausible competing-risk CIFs + interpretability views
   from HLA typing input. No real model; reproducible from inputs.
   ============================================================ */
(function () {
  'use strict';

  const LOCI = ['A', 'B', 'C', 'DRB1', 'DQB1'];
  // per-locus immunological weight (DRB1/DQB1 carry more GvHD weight)
  const W = { A: 0.9, B: 1.0, C: 0.7, DRB1: 1.25, DQB1: 1.1 };

  function hash(str) {
    let h = 2166136261 >>> 0;
    for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
    return h;
  }
  // pseudo-distance between two allele strings in [0,1]
  function alleleDist(a, b) {
    if (!a || !b) return 0;
    if (a === b) return 0;
    // field-aware: compare first field (e.g. 02 vs 07) more heavily
    const af = a.split(':'), bf = b.split(':');
    let d = 0;
    if (af[0] !== bf[0]) d += 0.7;
    if ((af[1] || '') !== (bf[1] || '')) d += 0.3;
    // jitter from hash so distinct alleles differ a touch
    const j = ((hash(a + '|' + b) % 1000) / 1000) * 0.12;
    return Math.min(1, d + j);
  }

  function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

  function predict(state) {
    // mismatch load per locus
    const perLocus = {};
    let totalMM = 0;
    LOCI.forEach(l => {
      const [d, r] = state[l] || ['', ''];
      const dist = alleleDist(d, r);
      perLocus[l] = dist;
      totalMM += dist * W[l];
    });
    const maxMM = LOCI.reduce((s, l) => s + W[l], 0); // ~4.95
    const load = totalMM / maxMM; // 0..1

    // baseline 5-yr incidences modulated by mismatch load
    const gvhd = clamp(0.10 + load * 0.55 + perLocus.DRB1 * 0.10, 0.04, 0.85);
    const relapse = clamp(0.34 - load * 0.20 + 0.04, 0.10, 0.62); // higher match → slightly higher relapse (GvL tradeoff)
    const trm = clamp(0.12 + load * 0.34, 0.05, 0.7);

    const fiveYr = { gvhd, relapse, trm };

    // build monotone CIF paths over 72 months (svg coords matching predict chart: x 48..624, y 220 bottom .. 14 top mapped 0..1)
    const X0 = 48, X1 = 624, Y0 = 220, Y1 = 14;
    function path(plateau, shape) {
      // shape: how fast it rises (gvhd fast, trm slow)
      const N = 40, pts = [];
      for (let i = 0; i <= N; i++) {
        const tt = i / N;
        // saturating curve
        const frac = 1 - Math.exp(-shape * tt);
        const norm = frac / (1 - Math.exp(-shape));
        const val = plateau * norm;
        const x = X0 + (X1 - X0) * tt;
        const y = Y0 + (Y1 - Y0) * val;
        pts.push([x, y]);
      }
      return 'M' + pts.map(p => p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' L ');
    }
    const paths = {
      gvhd: path(gvhd, 3.4),
      relapse: path(relapse, 2.2),
      trm: path(trm, 1.6)
    };

    // SHAP-style signed attribution toward relapse risk (more mismatch → lower relapse here)
    const attribution = LOCI.map(l => {
      const base = -perLocus[l] * W[l] * 0.5; // mismatch reduces relapse
      const jitter = ((hash('attr' + l + JSON.stringify(state[l])) % 200) / 1000) - 0.1;
      return { locus: l, v: clamp(base + jitter + 0.04, -0.6, 0.6) };
    });

    // 5x5 cross-attention matrix (donor rows × recipient cols), diagonal-heavy, mismatch perturbs
    const attention = LOCI.map((dl, i) => LOCI.map((rl, j) => {
      let v = i === j ? 0.62 : 0.18;
      v += ((hash(dl + rl + (state[dl] ? state[dl][0] : '') + (state[rl] ? state[rl][1] : '')) % 1000) / 1000) * 0.3;
      // higher mismatch at a locus heightens its attention
      v += (perLocus[dl] + perLocus[rl]) * 0.12;
      return clamp(v, 0, 1);
    }));

    // C-index style confidence (decreases with extreme mismatch)
    const confidence = clamp(0.84 - Math.abs(load - 0.4) * 0.18, 0.6, 0.86);

    return { fiveYr, paths, attribution, attention, load, confidence, perLocus };
  }

  window.CAPAModel = { predict, LOCI, alleleDist };
})();
