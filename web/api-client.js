/* CAPA API client — wraps the synthetic model with a real backend call.
   Tries POST /predict on the configured API URL; falls back to the
   deterministic synthetic model (model.js) when the backend is unreachable.   */
(function () {
  'use strict';

  const LOCI = ['A', 'B', 'C', 'DRB1', 'DQB1'];

  // Keep a reference to the synthetic engine before we add to the namespace
  const _syntheticPredict = window.CAPAModel.predict.bind(window.CAPAModel);

  // ── SVG path builder from CIF array ───────────────────────────────────────
  // Maps the API's 100-point CIF (0–730 days) to the same SVG coordinate
  // space used by predict.html / compare.html (x: 48–624, y: 220–14).
  function cifToPath(cif) {
    const X0 = 48, X1 = 624, Y0 = 220, Y1 = 14;
    const N = cif.length;
    const pts = [];
    for (let i = 0; i < N; i++) {
      const t  = i / (N - 1);
      const x  = (X0 + (X1 - X0) * t).toFixed(1);
      const y  = (Y0 + (Y1 - Y0) * Math.min(1, Math.max(0, cif[i]))).toFixed(1);
      pts.push(x + ' ' + y);
    }
    return 'M' + pts.join(' L ');
  }

  // ── Map API PredictionResponse → render format ────────────────────────────
  function mapApiResponse(data, state) {
    const paths = {
      gvhd:    cifToPath(data.gvhd.cumulative_incidence),
      relapse: cifToPath(data.relapse.cumulative_incidence),
      trm:     cifToPath(data.trm.cumulative_incidence),
    };
    const fiveYr = {
      gvhd:    data.gvhd.risk_score,
      relapse: data.relapse.risk_score,
      trm:     data.trm.risk_score,
    };
    // Attention weights from the real cross-attention layer; fall back synthetic
    const attention = (data.attention_weights && data.attention_weights.length)
      ? data.attention_weights
      : _syntheticPredict(state).attention;
    // Attribution (SHAP) is not returned by the API; keep synthetic
    const { attribution } = _syntheticPredict(state);

    return { fiveYr, paths, attribution, attention, fromAPI: true, modelVersion: data.model_version };
  }

  // ── Status badge ──────────────────────────────────────────────────────────
  let _badge = null;

  function badge() {
    if (_badge) return _badge;
    _badge = document.createElement('div');
    _badge.id = 'capa-status-badge';
    _badge.style.cssText = [
      'position:fixed', 'bottom:16px', 'left:16px', 'z-index:9998',
      'display:flex', 'align-items:center', 'gap:7px',
      'padding:5px 12px', 'border-radius:999px',
      'font-family:var(--mono,monospace)', 'font-size:11px',
      'border:1px solid var(--line,#e0e0e0)',
      'background:var(--paper,#fff)', 'color:var(--ink-3,#999)',
      'pointer-events:none', 'transition:opacity .3s',
      'letter-spacing:.04em'
    ].join(';');
    document.body.appendChild(_badge);
    return _badge;
  }

  function setStatus(live, version) {
    const dot = live ? '#22c55e' : 'var(--ink-3,#aaa)';
    const label = live
      ? 'Live · CAPA' + (version && version !== 'mock' ? ' · ' + version : '')
      : 'Synthetic demo';
    const el = badge();
    el.innerHTML =
      '<span style="width:7px;height:7px;border-radius:50%;background:' + dot +
      ';flex-shrink:0;display:inline-block"></span>' + label;
  }

  // ── Health check on page load ─────────────────────────────────────────────
  async function checkHealth() {
    const base = apiBase();
    try {
      const ac = new AbortController();
      setTimeout(() => ac.abort(), 3000);
      const res = await fetch(base + '/health', { signal: ac.signal });
      if (res.ok) {
        const d = await res.json();
        setStatus(d.ready, d.model_version);
      } else {
        setStatus(false);
      }
    } catch (_) {
      setStatus(false);
    }
  }

  function apiBase() {
    return (window.CAPA_CONFIG && window.CAPA_CONFIG.apiUrl) || 'http://localhost:8000';
  }

  // ── Core async predict ────────────────────────────────────────────────────
  async function predictAsync(state) {
    const base = apiBase();

    // Build API payload: locus inputs are bare "02:01" → prepend "A*" etc.
    const donor_hla = {}, recipient_hla = {};
    LOCI.forEach(function (l) {
      const pair = state[l] || ['', ''];
      if (pair[0]) donor_hla[l]     = l + '*' + pair[0];
      if (pair[1]) recipient_hla[l] = l + '*' + pair[1];
    });

    try {
      const ac = new AbortController();
      const tid = setTimeout(function () { ac.abort(); }, 12000);
      const res = await fetch(base + '/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ donor_hla: donor_hla, recipient_hla: recipient_hla }),
        signal: ac.signal,
      });
      clearTimeout(tid);

      if (!res.ok) throw new Error('HTTP ' + res.status);

      const data = await res.json();
      setStatus(true, data.model_version);
      return mapApiResponse(data, state);

    } catch (err) {
      if (err.name !== 'AbortError') {
        console.warn('[CAPA] Backend unavailable, using synthetic model:', err.message);
      }
      setStatus(false);
      return Object.assign(_syntheticPredict(state), { fromAPI: false });
    }
  }

  // ── Attach to global ──────────────────────────────────────────────────────
  window.CAPAModel.predictAsync = predictAsync;

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', checkHealth);
  } else {
    checkHealth();
  }
})();
