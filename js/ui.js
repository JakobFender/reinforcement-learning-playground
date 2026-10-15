// Shared UI helpers: controls, the per-tab layout, the run loop helper and the gridworld editor.
(function (root) {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const cssColor = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const store = {
    get(key, fallback) {
      try {
        const v = localStorage.getItem(key);
        return v ? JSON.parse(v) : fallback;
      } catch (e) {
        return fallback;
      }
    },
    set(key, value) {
      try {
        localStorage.setItem(key, JSON.stringify(value));
      } catch (e) {
        /* storage unavailable: ignore */
      }
    },
  };

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function fmt(v, digits = 3) {
    if (v == null || Number.isNaN(v)) return '–';
    if (v === Infinity) return '∞';
    if (v === -Infinity) return '−∞';
    if (v !== 0 && (Math.abs(v) >= 1e5 || Math.abs(v) < 1e-3)) return v.toExponential(1).replace('-', '−');
    return String(Number(v.toPrecision(digits))).replace('-', '−');
  }

  const fmtInt = (n) => Math.round(n).toLocaleString('en-US');

  function renderTex(target, tex) {
    if (root.katex) {
      try {
        root.katex.render(tex, target, { displayMode: true, throwOnError: false });
        target.classList.remove('raw');
        return;
      } catch (e) {
        /* fall through to raw text */
      }
    }
    target.textContent = tex;
    target.classList.add('raw');
  }

  function legendHtml(items) {
    return items
      .map(({ label, color, dashed, dot, box }) => {
        if (dot) return `<span><i class="dot" style="background:${color}"></i>${label}</span>`;
        if (box) return `<span><i class="box" style="background:${color}"></i>${label}</span>`;
        return `<span><i class="${dashed ? 'dashed' : ''}" style="background:${color};border-color:${color}"></i>${label}</span>`;
      })
      .join('');
  }

  function readoutHtml(items) {
    return items
      .map(({ label, value, title, accent }) => `<div${title ? ` title="${title}"` : ''}><dt>${label}</dt><dd${accent ? ' class="accent"' : ''}>${value}</dd></div>`)
      .join('');
  }

  // ---------------------------------------------------------------------------
  // Controls
  // ---------------------------------------------------------------------------

  // Slider + number input. With `log: true`, min/max are real values and the slider moves in log10 space.
  function slider(spec) {
    const wrap = el('div', 'param');
    const label = el('label', 'param-label', spec.label);
    const row = el('div', 'param-row');
    const range = document.createElement('input');
    range.type = 'range';
    const toSlider = spec.log ? (v) => Math.log10(v) : (v) => v;
    const fromSlider = spec.log ? (v) => Number(Math.pow(10, v).toPrecision(3)) : (v) => v;
    range.min = toSlider(spec.min);
    range.max = toSlider(spec.max);
    range.step = spec.log ? 0.01 : spec.step;
    range.setAttribute('aria-label', spec.label);
    const num = document.createElement('input');
    num.type = 'number';
    num.step = spec.inputStep || (spec.log ? 'any' : spec.step);
    num.setAttribute('aria-label', `${spec.label} value`);
    range.addEventListener('input', () => {
      spec.set(fromSlider(Number(range.value)));
      sync(false);
      if (spec.change) spec.change();
    });
    num.addEventListener('input', () => {
      const v = Number(num.value);
      if (num.value === '' || !isFinite(v)) return;
      spec.set(clamp(v, spec.min, spec.max));
      sync(true);
      if (spec.change) spec.change();
    });
    num.addEventListener('change', () => sync(false));
    row.append(range, num);
    wrap.append(label, row);
    function sync(skipNum) {
      const v = spec.get();
      range.value = toSlider(v);
      if (!skipNum) num.value = spec.display ? spec.display(v) : v;
      if (spec.labelFn) label.textContent = spec.labelFn(v);
    }
    sync(false);
    return { el: wrap, sync };
  }

  function segmented(options, get, set, ariaLabel, small) {
    const g = el('div', small ? 'segmented small' : 'segmented');
    g.setAttribute('role', 'radiogroup');
    g.setAttribute('aria-label', ariaLabel);
    const buttons = options.map(([value, text]) => {
      const b = el('button', null, text);
      b.type = 'button';
      b.dataset.value = value;
      b.addEventListener('click', () => {
        set(value);
        sync();
      });
      g.append(b);
      return b;
    });
    function sync() {
      for (const b of buttons) b.setAttribute('aria-pressed', String(b.dataset.value === String(get())));
    }
    sync();
    return { el: g, sync };
  }

  function select(labelText, options, get, set) {
    const wrap = el('label', 'field');
    if (labelText) wrap.append(el('span', null, labelText));
    const s = document.createElement('select');
    for (const [value, text] of options) {
      const o = document.createElement('option');
      o.value = value;
      o.textContent = text;
      s.append(o);
    }
    s.value = get();
    s.addEventListener('change', () => set(s.value));
    wrap.append(s);
    return { el: wrap, select: s, sync: () => (s.value = get()) };
  }

  function checkbox(labelText, get, set) {
    const wrap = el('label', 'check');
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.checked = get();
    box.addEventListener('change', () => set(box.checked));
    wrap.append(box, document.createTextNode(` ${labelText}`));
    return { el: wrap, sync: () => (box.checked = get()) };
  }

  function button(text, cls, onClick, title) {
    const b = el('button', cls, text);
    b.type = 'button';
    if (title) b.title = title;
    b.addEventListener('click', onClick);
    return b;
  }

  function heading(text, first) {
    return el('h3', first ? 'label first' : 'label', text);
  }

  // Run / pause + step buttons + reset. `extra` adds more ghost buttons (e.g. "Episode").
  function runControls(o) {
    const wrap = el('div', 'row-btns');
    let running = false;
    const run = button('Run', 'btn run', () => set(!running));
    wrap.append(run);
    const extra = [];
    if (o.onStep) extra.push(button(o.stepLabel || 'Step', 'btn ghost', () => { set(false); o.onStep(); }, o.stepTitle));
    for (const x of o.extra || []) extra.push(button(x.label, 'btn ghost', () => { set(false); x.fn(); }, x.title));
    extra.push(button('Reset', 'btn ghost', () => { set(false); o.onReset(); }));
    wrap.append(...extra);
    function set(v) {
      running = v;
      run.textContent = v ? 'Pause' : 'Run';
      if (o.onToggle) o.onToggle(v);
    }
    return {
      el: wrap,
      get running() {
        return running;
      },
      set,
      runButton: run,
    };
  }

  // Speed on a log scale: value is log10(steps per second).
  function speedSlider(o) {
    const state = { v: o.value };
    const s = slider({
      label: 'Speed',
      min: o.min,
      max: o.max,
      step: 0.01,
      get: () => state.v,
      set: (v) => (state.v = v),
      display: (v) => fmt(Math.pow(10, v), 2),
      labelFn: (v) => `Speed · ${fmtRate(Math.pow(10, v))} ${o.unit}/s`,
    });
    // The number box shows steps per second, not its log.
    const num = s.el.querySelector('input[type="number"]');
    const fresh = num.cloneNode(true);
    num.replaceWith(fresh);
    fresh.step = 'any';
    fresh.value = fmt(Math.pow(10, state.v), 2);
    fresh.addEventListener('change', () => {
      const v = Number(fresh.value);
      if (v > 0) {
        state.v = clamp(Math.log10(v), o.min, o.max);
        s.sync();
        fresh.value = fmt(Math.pow(10, state.v), 2);
      }
    });
    s.el.querySelector('input[type="range"]').addEventListener('input', () => (fresh.value = fmt(Math.pow(10, state.v), 2)));
    return { el: s.el, rate: () => Math.pow(10, state.v) };
  }

  function fmtRate(r) {
    if (r >= 1000) return `${fmt(r / 1000, 2)}k`;
    if (r >= 10) return String(Math.round(r));
    return fmt(r, 2);
  }

  // Runs `fn` at `rate` calls per second, within a per-frame time budget. Returns the number of calls made.
  function makeStepper() {
    let acc = 0;
    return {
      reset() {
        acc = 0;
      },
      advance(dt, rate, fn, budgetMs = 12) {
        acc += rate * dt;
        const n = Math.floor(acc);
        acc -= n;
        if (acc > 1) acc = 1;
        const t0 = performance.now();
        let done = 0;
        for (let i = 0; i < n; i++) {
          if (fn() === false) break;
          done++;
          if ((i & 15) === 15 && performance.now() - t0 > budgetMs) {
            acc = 0;
            break;
          }
        }
        return done;
      },
    };
  }

  // ---------------------------------------------------------------------------
  // Series helpers
  // ---------------------------------------------------------------------------

  // Trailing moving average.
  function smooth(arr, win) {
    const out = new Float64Array(arr.length);
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
      sum += arr[i];
      if (i >= win) sum -= arr[i - win];
      out[i] = sum / Math.min(i + 1, win);
    }
    return out;
  }

  // At most `max` points, evenly spaced (x is the 1-based index unless xs is given).
  function thin(ys, max = 600, xs = null) {
    const n = ys.length;
    const step = Math.max(1, Math.ceil(n / max));
    const ox = [];
    const oy = [];
    for (let i = 0; i < n; i += step) {
      ox.push(xs ? xs[i] : i + 1);
      oy.push(ys[i]);
    }
    if (n && (n - 1) % step) {
      ox.push(xs ? xs[n - 1] : n);
      oy.push(ys[n - 1]);
    }
    return { xs: ox, ys: oy };
  }

  function extent(arrays, pad = 0.05) {
    let lo = Infinity;
    let hi = -Infinity;
    for (const a of arrays) for (const v of a) if (isFinite(v)) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!isFinite(lo)) return [0, 1];
    if (hi - lo < 1e-9) {
      lo -= 1;
      hi += 1;
    }
    const p = (hi - lo) * pad;
    return [lo - p, hi + p];
  }

  // ---------------------------------------------------------------------------
  // Layout
  // ---------------------------------------------------------------------------

  // Builds the standard tab layout and returns handles to its parts.
  function layout(container, tab, info) {
    container.innerHTML = '';
    const ws = el('section', 'workspace');
    const controls = el('aside', 'controls');
    const label = el('p', 'label', `01 — ${tab.groupName}`);
    const head = el('div', 'algo-head');
    const h2 = el('h2');
    h2.innerHTML = info.title;
    head.append(h2, el('span', 'meta', info.ref));
    const desc = el('p', 'desc');
    desc.innerHTML = info.desc;
    controls.append(label, head, desc);
    const formulas = [];
    for (const tex of info.tex || []) {
      const f = el('div', 'formula');
      f.dataset.tex = tex;
      renderTex(f, tex);
      controls.append(f);
      formulas.push(f);
    }
    const body = el('div');
    controls.append(body);

    const card = el('div', 'plot-card');
    const toolbar = el('div', 'plot-toolbar');
    const tools = el('div', 'tools');
    const legend = el('div', 'legend');
    toolbar.append(tools, legend);
    card.append(toolbar);
    let strip = null;
    if (info.strip) {
      const sw = el('div', 'plot-wrap strip');
      strip = el('canvas');
      sw.append(strip);
      card.append(sw);
    }
    const wrap = el('div', 'plot-wrap main');
    const canvas = el('canvas');
    if (info.ariaLabel) canvas.setAttribute('aria-label', info.ariaLabel);
    wrap.append(canvas);
    const readout = el('dl', 'readout');
    card.append(wrap, readout);
    ws.append(controls, card);

    const under = el('section', 'under');
    const uh = el('header', 'under-head');
    uh.append(el('p', 'label', '02 — Under the hood'));
    const uh2 = el('h2');
    uh2.innerHTML = info.underTitle || 'What it <em>learned</em>';
    uh.append(uh2);
    const grid = el('div', info.minis === 2 ? 'mini-grid two' : 'mini-grid');
    under.append(uh, grid);
    container.append(ws, under);

    return {
      controls: body,
      tools,
      legend,
      canvas,
      strip,
      wrap,
      readout,
      grid,
      retex: () => formulas.forEach((f) => renderTex(f, f.dataset.tex)),
      setReadout: (items) => (readout.innerHTML = readoutHtml(items)),
      setLegend: (items) => (legend.innerHTML = legendHtml(items)),
    };
  }

  // A small captioned chart in the "under the hood" grid.
  function mini(grid, title, note, opts = {}) {
    const fig = el('figure', 'mini');
    const cap = el('figcaption');
    const t = el('span', null, title);
    const meta = el('span', 'meta', opts.meta || '');
    cap.append(t, meta);
    const wrap = el('div', 'plot-wrap mini-wrap');
    const canvas = el('canvas');
    wrap.append(canvas);
    const n = el('p', 'mini-note');
    n.innerHTML = note || '';
    fig.append(cap, wrap, n);
    grid.append(fig);
    return { fig, cap, title: t, meta, wrap, canvas, note: n };
  }

  // ---------------------------------------------------------------------------
  // Gridworld environment controls (preset, γ, slip, editing brush)
  // ---------------------------------------------------------------------------

  const BRUSHES = [
    ['#', 'Wall'],
    ['G', 'Goal'],
    ['X', 'Cliff'],
    ['S', 'Start'],
    ['.', 'Erase'],
  ];

  // state: { preset, gamma, slip, env }. onChange(kind) is called with 'env' or 'gamma'.
  function gridControls(container, state, onChange, opts = {}) {
    const RL = root.RL;
    const presetOptions = Object.entries(RL.gridPresets).map(([k, p]) => [k, `${p.name} · ${p.ref}`]);
    const sel = select('Layout', presetOptions, () => state.preset, (v) => {
      state.preset = v;
      state.env = RL.GridWorld.fromPreset(v, { slip: state.slip });
      if (opts.presetGamma !== false) {
        state.gamma = RL.gridPresets[v].gamma;
        gammaSlider.sync();
      }
      onChange('env');
    });
    container.append(sel.el);
    const params = el('div', 'params');
    params.style.marginTop = 'var(--space-3)';
    const gammaSlider = slider({
      label: 'Discount γ',
      min: 0.5,
      max: 1,
      step: 0.01,
      get: () => state.gamma,
      set: (v) => (state.gamma = v),
      labelFn: (v) => `Discount γ${v === 1 ? ' · undiscounted' : ''}`,
      change: () => onChange('gamma'),
    });
    const slipSlider = slider({
      label: 'Slip probability',
      min: 0,
      max: 0.5,
      step: 0.05,
      get: () => state.slip,
      set: (v) => {
        state.slip = v;
        state.env.slip = v;
      },
      labelFn: (v) => `Slip probability${v ? ' · random action ' + Math.round(v * 100) + '% of the time' : ' · deterministic'}`,
      change: () => onChange('slip'),
    });
    params.append(gammaSlider.el, slipSlider.el);
    container.append(params);

    const brushWrap = el('div', 'brush');
    brushWrap.append(el('span', 'meta', 'Click or drag on the grid to paint'));
    state.brush = state.brush || '#';
    const brush = segmented(BRUSHES, () => state.brush, (v) => (state.brush = v), 'Brush', true);
    brushWrap.append(brush.el);
    container.append(brushWrap);
    return { sync: () => { sel.sync(); gammaSlider.sync(); slipSlider.sync(); } };
  }

  // Painting on the grid canvas. getGeom() returns the latest gridGeometry. onPaint() after each change.
  function attachGridEditor(canvas, state, getGeom, onPaint, onHover) {
    canvas.classList.add('editable');
    let painting = false;
    let last = null;
    function paint(e) {
      const g = getGeom();
      if (!g) return;
      const r = canvas.getBoundingClientRect();
      const cell = g.cellAt(e.clientX - r.left, e.clientY - r.top);
      if (!cell || cell.s === last) return;
      last = cell.s;
      const env = state.env;
      if (env.cell(cell.s) === state.brush) return;
      // Keep at least one non-wall cell.
      env.setCell(cell.r, cell.c, state.brush);
      onPaint();
    }
    canvas.addEventListener('pointerdown', (e) => {
      painting = true;
      last = null;
      canvas.setPointerCapture(e.pointerId);
      paint(e);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (painting) paint(e);
      if (onHover) {
        const g = getGeom();
        if (!g) return;
        const r = canvas.getBoundingClientRect();
        const cell = g.cellAt(e.clientX - r.left, e.clientY - r.top);
        onHover(cell ? cell.s : null);
      }
    });
    const stop = () => {
      painting = false;
      last = null;
    };
    canvas.addEventListener('pointerup', stop);
    canvas.addEventListener('pointercancel', stop);
    canvas.addEventListener('pointerleave', () => onHover && onHover(null));
  }

  // ---------------------------------------------------------------------------
  // Theme
  // ---------------------------------------------------------------------------

  const themeListeners = [];
  function onTheme(fn) {
    themeListeners.push(fn);
  }
  function initTheme() {
    const saved = store.get('rlp-theme', null);
    if (saved) document.documentElement.setAttribute('data-theme', saved);
    $('theme-toggle').addEventListener('click', () => {
      const current =
        document.documentElement.getAttribute('data-theme') ||
        (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      const next = current === 'dark' ? 'light' : 'dark';
      store.set('rlp-theme', next);
      document.documentElement.setAttribute('data-theme', next);
      themeListeners.forEach((fn) => fn());
    });
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => themeListeners.forEach((fn) => fn()));
  }

  root.UI = {
    $,
    el,
    clamp,
    cssColor,
    store,
    fmt,
    fmtInt,
    renderTex,
    legendHtml,
    readoutHtml,
    slider,
    segmented,
    select,
    checkbox,
    button,
    heading,
    runControls,
    speedSlider,
    makeStepper,
    smooth,
    thin,
    extent,
    layout,
    mini,
    gridControls,
    attachGridEditor,
    onTheme,
    initTheme,
  };
  root.Views = root.Views || {};
})(typeof globalThis !== 'undefined' ? globalThis : this);
