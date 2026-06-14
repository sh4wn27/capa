/* ============================================================
   CAPA — shared site JS
   - scroll reveal
   - mobile nav
   - lightweight vanilla Tweaks panel (accent + display font)
   Applies persisted tweaks before paint where possible.
   ============================================================ */
(function () {
  'use strict';

  /* ---------- tweak presets ---------- */
  const ACCENTS = {
    clinical: { name: 'Clinical blue', accent: 'oklch(0.48 0.12 255)', ink: 'oklch(0.40 0.13 255)', soft: 'oklch(0.48 0.12 255 / 0.10)' },
    oxblood:  { name: 'Oxblood',       accent: 'oklch(0.46 0.13 25)',  ink: 'oklch(0.39 0.14 25)',  soft: 'oklch(0.46 0.13 25 / 0.10)' },
    pine:     { name: 'Pine',          accent: 'oklch(0.46 0.09 165)', ink: 'oklch(0.39 0.10 165)', soft: 'oklch(0.46 0.09 165 / 0.10)' },
    graphite: { name: 'Graphite',      accent: 'oklch(0.40 0.012 260)',ink: 'oklch(0.30 0.012 260)',soft: 'oklch(0.40 0.012 260 / 0.10)' }
  };
  const FONTS = {
    newsreader: { name: 'Newsreader',  stack: "'Newsreader', Georgia, serif" },
    fraunces:   { name: 'Spectral',    stack: "'Spectral', Georgia, serif" },
    grotesk:    { name: 'Grotesk',     stack: "'Hanken Grotesk', system-ui, sans-serif" }
  };

  const DEFAULTS = { accent: 'oxblood', display: 'newsreader' };
  const STORE = 'capa.tweaks.v1';

  function load() {
    try { return Object.assign({}, DEFAULTS, JSON.parse(localStorage.getItem(STORE) || '{}')); }
    catch (e) { return Object.assign({}, DEFAULTS); }
  }
  function save(t) { try { localStorage.setItem(STORE, JSON.stringify(t)); } catch (e) {} }

  function apply(t) {
    const a = ACCENTS[t.accent] || ACCENTS.clinical;
    const f = FONTS[t.display] || FONTS.newsreader;
    const r = document.documentElement.style;
    r.setProperty('--accent', a.accent);
    r.setProperty('--accent-ink', a.ink);
    r.setProperty('--accent-2', a.soft);
    r.setProperty('--serif', f.stack);
  }

  let tweaks = load();
  apply(tweaks);

  /* ---------- DOM ready ---------- */
  function ready(fn) {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }

  ready(function () {
    /* scroll reveal */
    const io = 'IntersectionObserver' in window
      ? new IntersectionObserver((entries) => {
          entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
        }, { rootMargin: '0px 0px -8% 0px', threshold: 0.08 })
      : null;
    document.querySelectorAll('.reveal').forEach((el) => { if (io) io.observe(el); else el.classList.add('in'); });

    /* mobile nav */
    const tgl = document.querySelector('[data-nav-toggle]');
    const menu = document.querySelector('[data-nav-menu]');
    if (tgl && menu) tgl.addEventListener('click', () => menu.classList.toggle('open'));

    buildTweaksPanel();
    initBetaNotice();
  });

  /* ---------- tweaks panel (vanilla, host-protocol aware) ---------- */
  function buildTweaksPanel() {
    const style = document.createElement('style');
    style.textContent = `
      .ctk{position:fixed;right:16px;bottom:16px;z-index:2147483646;width:248px;
        background:oklch(0.989 0.004 85 / 0.86);color:var(--ink);
        -webkit-backdrop-filter:blur(20px) saturate(150%);backdrop-filter:blur(20px) saturate(150%);
        border:1px solid var(--line);border-radius:12px;
        box-shadow:0 16px 44px rgba(0,0,0,.16);
        font-family:var(--sans);font-size:12px;overflow:hidden;display:none}
      .ctk.open{display:block}
      .ctk-hd{display:flex;align-items:center;justify-content:space-between;
        padding:11px 10px 11px 14px;cursor:move;border-bottom:1px solid var(--line-2)}
      .ctk-hd b{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;font-weight:500;color:var(--ink-2)}
      .ctk-x{border:0;background:transparent;color:var(--ink-3);width:22px;height:22px;border-radius:6px;cursor:pointer;font-size:13px}
      .ctk-x:hover{background:var(--paper-3);color:var(--ink)}
      .ctk-body{padding:14px;display:flex;flex-direction:column;gap:16px}
      .ctk-sec{display:flex;flex-direction:column;gap:8px}
      .ctk-lbl{font-family:var(--mono);font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}
      .ctk-chips{display:flex;gap:7px}
      .ctk-chip{flex:1;height:30px;border-radius:7px;border:1px solid var(--line);cursor:pointer;position:relative;padding:0;transition:transform .12s}
      .ctk-chip:hover{transform:translateY(-1px)}
      .ctk-chip[data-on="1"]{box-shadow:0 0 0 2px var(--ink)}
      .ctk-seg{display:flex;gap:0;border:1px solid var(--line);border-radius:7px;overflow:hidden}
      .ctk-seg button{flex:1;border:0;background:transparent;color:var(--ink-2);font-family:var(--sans);font-size:11.5px;font-weight:500;padding:7px 4px;cursor:pointer;border-right:1px solid var(--line-2)}
      .ctk-seg button:last-child{border-right:0}
      .ctk-seg button[data-on="1"]{background:var(--ink);color:var(--paper)}
    `;
    document.head.appendChild(style);

    const panel = document.createElement('div');
    panel.className = 'ctk';
    panel.setAttribute('data-omelette-chrome', '');
    panel.innerHTML = `
      <div class="ctk-hd" data-drag><b>Tweaks</b><button class="ctk-x" aria-label="Close">✕</button></div>
      <div class="ctk-body">
        <div class="ctk-sec">
          <div class="ctk-lbl">Accent</div>
          <div class="ctk-chips" data-accent></div>
        </div>
        <div class="ctk-sec">
          <div class="ctk-lbl">Display type</div>
          <div class="ctk-seg" data-font></div>
        </div>
      </div>`;
    document.body.appendChild(panel);

    const accWrap = panel.querySelector('[data-accent]');
    Object.keys(ACCENTS).forEach((k) => {
      const b = document.createElement('button');
      b.className = 'ctk-chip';
      b.style.background = ACCENTS[k].accent;
      b.title = ACCENTS[k].name;
      b.dataset.k = k;
      b.addEventListener('click', () => { tweaks.accent = k; commit(); });
      accWrap.appendChild(b);
    });
    const fontWrap = panel.querySelector('[data-font]');
    Object.keys(FONTS).forEach((k) => {
      const b = document.createElement('button');
      b.textContent = FONTS[k].name;
      b.dataset.k = k;
      b.style.fontFamily = FONTS[k].stack;
      b.addEventListener('click', () => { tweaks.display = k; commit(); });
      fontWrap.appendChild(b);
    });

    function syncUI() {
      accWrap.querySelectorAll('.ctk-chip').forEach((c) => c.dataset.on = (c.dataset.k === tweaks.accent ? '1' : '0'));
      fontWrap.querySelectorAll('button').forEach((c) => c.dataset.on = (c.dataset.k === tweaks.display ? '1' : '0'));
    }
    function commit() { apply(tweaks); save(tweaks); syncUI();
      window.parent.postMessage({ type: '__edit_mode_set_keys', edits: tweaks }, '*'); }
    syncUI();

    /* close */
    panel.querySelector('.ctk-x').addEventListener('click', () => {
      panel.classList.remove('open');
      window.parent.postMessage({ type: '__edit_mode_dismissed' }, '*');
    });

    /* drag */
    const hd = panel.querySelector('[data-drag]');
    hd.addEventListener('mousedown', (e) => {
      if (e.target.classList.contains('ctk-x')) return;
      const r = panel.getBoundingClientRect();
      let sr = window.innerWidth - r.right, sb = window.innerHeight - r.bottom;
      const sx = e.clientX, sy = e.clientY;
      const move = (ev) => {
        panel.style.right = Math.max(8, sr - (ev.clientX - sx)) + 'px';
        panel.style.bottom = Math.max(8, sb - (ev.clientY - sy)) + 'px';
      };
      const up = () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up); };
      window.addEventListener('mousemove', move); window.addEventListener('mouseup', up);
    });

    /* host protocol */
    window.addEventListener('message', (e) => {
      const t = e && e.data && e.data.type;
      if (t === '__activate_edit_mode') panel.classList.add('open');
      else if (t === '__deactivate_edit_mode') panel.classList.remove('open');
    });
    window.parent.postMessage({ type: '__edit_mode_available' }, '*');
  }

  /* ---------- beta notice ---------- */
  function initBetaNotice() {
    const KEY = 'capa.beta-notice.v1';
    if (localStorage.getItem(KEY)) return;

    const overlay = document.createElement('div');
    overlay.className = 'beta-overlay';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-labelledby', 'beta-title');
    overlay.innerHTML = `
      <div class="beta-modal">
        <div class="beta-eyebrow">Beta &middot; Active Development</div>
        <h3 id="beta-title" class="serif">CAPA is under active development</h3>
        <p>This tool is in <strong>closed beta</strong> and may not perform as described. Features are incomplete and models are still being validated — treat all outputs as experimental.</p>
        <p>By continuing, you acknowledge CAPA is a research prototype and <strong>not cleared for clinical use</strong>.</p>
        <div class="beta-actions">
          <button class="btn btn-primary" id="beta-ack">Understood — continue to beta</button>
          <a href="about.html" class="btn btn-ghost">Learn more</a>
        </div>
      </div>`;
    document.body.appendChild(overlay);

    requestAnimationFrame(() => requestAnimationFrame(() => overlay.classList.add('visible')));

    function dismiss() {
      try { localStorage.setItem(KEY, '1'); } catch (e) {}
      overlay.classList.remove('visible');
      setTimeout(() => overlay.remove(), 300);
    }

    overlay.querySelector('#beta-ack').addEventListener('click', dismiss);
    overlay.addEventListener('click', (e) => { if (e.target === overlay) dismiss(); });
    document.addEventListener('keydown', function esc(e) {
      if (e.key === 'Escape') { dismiss(); document.removeEventListener('keydown', esc); }
    });
  }
})();
