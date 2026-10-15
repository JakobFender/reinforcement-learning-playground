// Canvas renderers that are not x/y charts: gridworlds, the cart-pole, policy heatmaps and the random-walk chain.
(function (root) {
  'use strict';

  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function withAlpha(color, alpha) {
    if (color.startsWith('#') && color.length === 7) {
      const r = parseInt(color.slice(1, 3), 16);
      const g = parseInt(color.slice(3, 5), 16);
      const b = parseInt(color.slice(5, 7), 16);
      return `rgba(${r},${g},${b},${alpha})`;
    }
    return color;
  }

  function palette() {
    return {
      bg: cssVar('--color-bg'),
      text: cssVar('--color-text'),
      muted: cssVar('--color-muted'),
      accent: cssVar('--color-accent'),
      line: cssVar('--color-line'),
      surface: cssVar('--color-surface'),
      grid: cssVar('--grid'),
      wall: cssVar('--wall'),
    };
  }

  const MONO = "'Martian Mono', ui-monospace, monospace";

  // A canvas that keeps its backing store in sync with its CSS size and redraws on resize.
  class Surface {
    constructor(canvas, draw) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.drawFn = draw;
      new ResizeObserver(() => this.draw()).observe(canvas.parentElement);
      if (document.fonts) document.fonts.ready.then(() => this.draw());
    }

    draw() {
      const canvas = this.canvas;
      const parent = canvas.parentElement;
      const dpr = window.devicePixelRatio || 1;
      const w = parent.clientWidth;
      const h = parent.clientHeight;
      if (!w || !h) return;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
        canvas.style.width = `${w}px`;
        canvas.style.height = `${h}px`;
      }
      this.width = w;
      this.height = h;
      const ctx = this.ctx;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      this.drawFn(ctx, w, h);
    }

    local(e) {
      const r = this.canvas.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    }
  }

  // ---------------------------------------------------------------------------
  // Gridworld
  // ---------------------------------------------------------------------------

  // Layout of the grid inside a w × h box; also used to map pointer positions to cells.
  function gridGeometry(env, w, h, pad = 8) {
    const size = Math.max(4, Math.floor(Math.min((w - 2 * pad) / env.cols, (h - 2 * pad) / env.rows)));
    const ox = Math.round((w - size * env.cols) / 2);
    const oy = Math.round((h - size * env.rows) / 2);
    return {
      size,
      ox,
      oy,
      center: (s) => [ox + (s % env.cols) * size + size / 2, oy + Math.floor(s / env.cols) * size + size / 2],
      cellAt(x, y) {
        const c = Math.floor((x - ox) / size);
        const r = Math.floor((y - oy) / size);
        if (r < 0 || c < 0 || r >= env.rows || c >= env.cols) return null;
        return { r, c, s: r * env.cols + c };
      },
    };
  }

  function rangeOf(values, env, mask) {
    let lo = Infinity;
    let hi = -Infinity;
    for (let s = 0; s < env.nS; s++) {
      if (!env.isState(s) || env.isTerminal(s) || (mask && !mask[s])) continue;
      const v = values[s];
      if (!isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!isFinite(lo)) return [0, 1];
    if (hi - lo < 1e-9) return [lo - 1, hi + 1];
    return [lo, hi];
  }

  function fmtCell(v) {
    if (!isFinite(v)) return '';
    const a = Math.abs(v);
    if (a >= 1000) return v.toFixed(0);
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    return v.toFixed(2);
  }

  // opts: values, qs (per-action values → triangles), policy (probabilities per state), heat (visit counts),
  // agent, trail, path, hover, numbers, rootWeights { s, w[4] }.
  function drawGrid(ctx, w, h, env, opts = {}) {
    const P = palette();
    const g = gridGeometry(env, w, h);
    const { size, ox, oy } = g;
    const A = root.RL.ACTIONS;

    let range = null;
    if (opts.values) range = opts.range || rangeOf(opts.values, env, opts.mask);
    let qRange = null;
    if (opts.qs) {
      let lo = Infinity;
      let hi = -Infinity;
      for (let s = 0; s < env.nS; s++) {
        if (!env.isState(s) || env.isTerminal(s) || (opts.mask && !opts.mask[s])) continue;
        for (const q of opts.qs[s]) {
          if (q < lo) lo = q;
          if (q > hi) hi = q;
        }
      }
      qRange = isFinite(lo) && hi > lo ? [lo, hi] : [lo - 1, lo + 1];
    }
    let heatMax = 0;
    if (opts.heat) for (let s = 0; s < env.nS; s++) heatMax = Math.max(heatMax, opts.heat[s]);
    const norm = (v, rg) => (rg[1] > rg[0] ? (v - rg[0]) / (rg[1] - rg[0]) : 0.5);

    for (let s = 0; s < env.nS; s++) {
      const r = Math.floor(s / env.cols);
      const c = s % env.cols;
      const x = ox + c * size;
      const y = oy + r * size;
      const ch = env.cell(s);
      if (ch === '#') {
        ctx.fillStyle = P.wall;
        ctx.fillRect(x, y, size, size);
        continue;
      }
      if (ch === 'X') {
        ctx.save();
        ctx.beginPath();
        ctx.rect(x, y, size, size);
        ctx.clip();
        ctx.strokeStyle = P.muted;
        ctx.lineWidth = 1;
        for (let k = -size; k < size; k += 6) {
          ctx.beginPath();
          ctx.moveTo(x + k, y + size);
          ctx.lineTo(x + k + size, y);
          ctx.stroke();
        }
        ctx.restore();
        continue;
      }
      if (opts.mask && !opts.mask[s]) {
        // Never visited: nothing learned yet, leave blank.
      } else if (opts.qs && !env.isTerminal(s)) {
        const cx = x + size / 2;
        const cy = y + size / 2;
        const corners = [
          [x, y, x + size, y],
          [x + size, y, x + size, y + size],
          [x + size, y + size, x, y + size],
          [x, y + size, x, y],
        ];
        for (let a = 0; a < 4; a++) {
          const t = norm(opts.qs[s][a], qRange);
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(corners[a][0], corners[a][1]);
          ctx.lineTo(corners[a][2], corners[a][3]);
          ctx.closePath();
          ctx.fillStyle = withAlpha(P.accent, 0.04 + 0.5 * t * t);
          ctx.fill();
        }
      } else if (opts.values && !env.isTerminal(s)) {
        const t = norm(opts.values[s], range);
        ctx.fillStyle = withAlpha(P.accent, 0.04 + 0.5 * t * t);
        ctx.fillRect(x, y, size, size);
      } else if (opts.heat && heatMax > 0) {
        const t = Math.log1p(opts.heat[s]) / Math.log1p(heatMax);
        ctx.fillStyle = withAlpha(P.accent, 0.55 * t);
        ctx.fillRect(x, y, size, size);
      }
      if (ch === 'G') {
        ctx.strokeStyle = P.accent;
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 3, y + 3, size - 6, size - 6);
      }
    }

    // Grid lines
    ctx.strokeStyle = P.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let c = 0; c <= env.cols; c++) {
      const x = Math.round(ox + c * size) + 0.5;
      ctx.moveTo(x, oy);
      ctx.lineTo(x, oy + env.rows * size);
    }
    for (let r = 0; r <= env.rows; r++) {
      const y = Math.round(oy + r * size) + 0.5;
      ctx.moveTo(ox, y);
      ctx.lineTo(ox + env.cols * size, y);
    }
    ctx.stroke();
    if (opts.qs && size >= 24) {
      ctx.strokeStyle = withAlpha(P.line, 0.6) || P.line;
      ctx.beginPath();
      for (let s = 0; s < env.nS; s++) {
        if (!env.isState(s) || env.isTerminal(s) || (opts.mask && !opts.mask[s])) continue;
        const x = ox + (s % env.cols) * size;
        const y = oy + Math.floor(s / env.cols) * size;
        ctx.moveTo(x, y);
        ctx.lineTo(x + size, y + size);
        ctx.moveTo(x + size, y);
        ctx.lineTo(x, y + size);
      }
      ctx.stroke();
    }

    // Wind strengths under the grid
    if (env.wind.some((v) => v)) {
      ctx.fillStyle = P.muted;
      ctx.font = `11px ${MONO}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      for (let c = 0; c < env.cols; c++) {
        if (!env.wind[c]) continue;
        ctx.fillText(`↑${env.wind[c]}`, ox + c * size + size / 2, oy + env.rows * size + 2);
      }
    }

    // Labels: S, G, numbers
    const fs = Math.max(9, Math.min(12, Math.floor(size / 4.2)));
    ctx.font = `${fs}px ${MONO}`;
    for (let s = 0; s < env.nS; s++) {
      const ch = env.cell(s);
      const x = ox + (s % env.cols) * size;
      const y = oy + Math.floor(s / env.cols) * size;
      if (ch === 'S' || ch === 'G') {
        ctx.fillStyle = ch === 'G' ? P.accent : P.muted;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(ch, x + 5, y + 4);
      }
      if (opts.numbers && size >= 40 && env.isState(s) && !env.isTerminal(s) && !(opts.mask && !opts.mask[s])) {
        const v = opts.numbers[s];
        ctx.fillStyle = P.text;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(fmtCell(v), x + size / 2, y + size - 3);
      }
    }

    // Policy arrows
    if (opts.policy) {
      ctx.strokeStyle = P.text;
      ctx.fillStyle = P.text;
      ctx.lineWidth = Math.max(1, Math.min(2, size / 24));
      const len = size * (opts.numbers && size >= 40 ? 0.26 : 0.34);
      for (let s = 0; s < env.nS; s++) {
        if (!env.isState(s) || env.isTerminal(s)) continue;
        const p = opts.policy[s];
        if (!p) continue;
        const [cx0, cy0] = g.center(s);
        const cy = opts.numbers && size >= 40 ? cy0 - size * 0.1 : cy0;
        const pmax = Math.max(...p);
        for (let a = 0; a < 4; a++) {
          if (p[a] < 1e-6 || p[a] < pmax - 1e-6) continue;
          const dx = A[a].dc;
          const dy = A[a].dr;
          const ex = cx0 + dx * len;
          const ey = cy + dy * len;
          ctx.beginPath();
          ctx.moveTo(cx0, cy);
          ctx.lineTo(ex, ey);
          ctx.stroke();
          const hs = Math.max(3, size * 0.09);
          ctx.beginPath();
          ctx.moveTo(ex + dx * hs * 0.4, ey + dy * hs * 0.4);
          ctx.lineTo(ex - dx * hs + dy * hs * 0.7, ey - dy * hs + dx * hs * 0.7);
          ctx.lineTo(ex - dx * hs - dy * hs * 0.7, ey - dy * hs - dx * hs * 0.7);
          ctx.closePath();
          ctx.fill();
        }
      }
    }

    // Greedy path (dashed)
    if (opts.path && opts.path.length > 1) {
      ctx.strokeStyle = P.text;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      opts.path.forEach((s, i) => {
        const [x, y] = g.center(s);
        if (i) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Trail of the current episode
    if (opts.trail && opts.trail.length > 1) {
      ctx.strokeStyle = withAlpha(P.accent, 0.9);
      ctx.lineWidth = Math.max(2, size / 14);
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      let prev = null;
      for (const s of opts.trail) {
        const [x, y] = g.center(s);
        // Break the line on teleports (falling off the cliff back to the start).
        if (prev && Math.abs(prev[0] - x) + Math.abs(prev[1] - y) > size * 2.5) ctx.moveTo(x, y);
        else if (prev) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
        prev = [x, y];
      }
      ctx.stroke();
    }

    // Root action weights (MCTS visit shares)
    if (opts.rootWeights) {
      const { s, w: wts } = opts.rootWeights;
      const [cx, cy] = g.center(s);
      const tot = wts.reduce((a, b) => a + b, 0) || 1;
      ctx.strokeStyle = P.accent;
      ctx.lineCap = 'round';
      for (let a = 0; a < 4; a++) {
        const f = wts[a] / tot;
        if (f <= 0) continue;
        ctx.lineWidth = Math.max(2, size / 9);
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + A[a].dc * size * (0.2 + 0.8 * f), cy + A[a].dr * size * (0.2 + 0.8 * f));
        ctx.stroke();
      }
    }

    // Agent
    if (opts.agent != null) {
      const [x, y] = g.center(opts.agent);
      ctx.beginPath();
      ctx.arc(x, y, Math.max(4, size * 0.2), 0, 2 * Math.PI);
      ctx.fillStyle = P.text;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = P.bg;
      ctx.stroke();
    }

    // Selected cell
    if (opts.selected != null) {
      const x = ox + (opts.selected % env.cols) * size;
      const y = oy + Math.floor(opts.selected / env.cols) * size;
      ctx.strokeStyle = P.text;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x + 1.25, y + 1.25, size - 2.5, size - 2.5);
    }

    // Hover
    if (opts.hover != null && opts.hover !== opts.selected) {
      const x = ox + (opts.hover % env.cols) * size;
      const y = oy + Math.floor(opts.hover / env.cols) * size;
      ctx.strokeStyle = P.muted;
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
    }
    return g;
  }

  // ---------------------------------------------------------------------------
  // Cart-pole
  // ---------------------------------------------------------------------------

  function drawCartPole(ctx, w, h, state, opts = {}) {
    const P = palette();
    const C = root.RL.CARTPOLE;
    const margin = 24;
    const scale = (w - 2 * margin) / (2 * C.xLimit + 0.6);
    const cx = w / 2;
    const groundY = Math.round(h * 0.72) + 0.5;
    const [x, , theta] = state;

    // Track and limits
    ctx.strokeStyle = P.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(margin, groundY);
    ctx.lineTo(w - margin, groundY);
    ctx.stroke();
    ctx.font = `11px ${MONO}`;
    ctx.fillStyle = P.muted;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const lim of [-C.xLimit, 0, C.xLimit]) {
      const px = Math.round(cx + lim * scale) + 0.5;
      ctx.strokeStyle = lim ? P.muted : P.line;
      ctx.setLineDash(lim ? [3, 3] : []);
      ctx.beginPath();
      ctx.moveTo(px, groundY - (lim ? 18 : 6));
      ctx.lineTo(px, groundY + 6);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillText(lim ? `${lim > 0 ? '+' : '−'}${C.xLimit} m` : '0', px, groundY + 10);
    }

    // Pole length on screen: 2 × half-length, exaggerated slightly so it reads at small sizes.
    const cartW = Math.max(40, scale * 0.5);
    const cartH = cartW * 0.5;
    const px = cx + x * scale;
    const top = groundY - cartH - 6;
    const poleLen = Math.min(h * 0.6, scale * 2 * C.halfLength * 1.6);

    // Pole (drawn first so the cart covers its foot)
    const fallen = Math.abs(theta) > C.thetaLimit;
    ctx.strokeStyle = fallen ? P.muted : P.accent;
    ctx.lineWidth = Math.max(6, cartW * 0.12);
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(px, top + 4);
    ctx.lineTo(px + Math.sin(theta) * poleLen, top + 4 - Math.cos(theta) * poleLen);
    ctx.stroke();
    ctx.lineCap = 'butt';

    // Cart
    ctx.fillStyle = P.text;
    ctx.fillRect(px - cartW / 2, top, cartW, cartH);
    ctx.fillStyle = P.bg;
    for (const dx of [-cartW * 0.28, cartW * 0.28]) {
      ctx.beginPath();
      ctx.arc(px + dx, top + cartH + 3, 4, 0, 2 * Math.PI);
      ctx.fillStyle = P.text;
      ctx.fill();
    }
    ctx.beginPath();
    ctx.arc(px, top + 4, 4, 0, 2 * Math.PI);
    ctx.fillStyle = P.bg;
    ctx.fill();

    // Push arrow
    if (opts.action != null) {
      const dir = opts.action === 1 ? 1 : -1;
      const ax = px - dir * (cartW / 2 + 8);
      const ay = top + cartH / 2;
      ctx.strokeStyle = P.text;
      ctx.fillStyle = P.text;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(ax - dir * 22, ay);
      ctx.lineTo(ax - dir * 4, ay);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(ax - dir * 8, ay - 5);
      ctx.lineTo(ax - dir * 8, ay + 5);
      ctx.closePath();
      ctx.fill();
    }

    // Caption
    if (opts.caption) {
      ctx.font = `12px ${MONO}`;
      ctx.fillStyle = P.muted;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      opts.caption.forEach((line, i) => ctx.fillText(line, margin, 8 + i * 18));
    }
  }

  // ---------------------------------------------------------------------------
  // Heatmap over a 2-D slice (used for cart-pole policy/value maps)
  // ---------------------------------------------------------------------------

  // values: Float64Array(nx * ny), row-major with y increasing upwards. colorOf(v) → css color.
  function drawHeatmap(ctx, w, h, map, opts) {
    const P = palette();
    const m = { left: 48, right: 8, top: 8, bottom: 32 };
    const pw = w - m.left - m.right;
    const ph = h - m.top - m.bottom;
    const { nx, ny, values } = map;
    const cw = pw / nx;
    const chh = ph / ny;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        ctx.fillStyle = opts.colorOf(values[j * nx + i]);
        ctx.fillRect(m.left + i * cw, m.top + (ny - 1 - j) * chh, Math.ceil(cw), Math.ceil(chh));
      }
    }
    ctx.strokeStyle = P.line;
    ctx.lineWidth = 1;
    ctx.strokeRect(m.left + 0.5, m.top + 0.5, pw - 1, ph - 1);
    // Axes labels
    ctx.font = `11px ${MONO}`;
    ctx.fillStyle = P.muted;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const [x0, x1] = opts.xRange;
    const [y0, y1] = opts.yRange;
    [x0, (x0 + x1) / 2, x1].forEach((v, k) => ctx.fillText(opts.xFmt(v), m.left + (k * pw) / 2, m.top + ph + 6));
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    [y0, (y0 + y1) / 2, y1].forEach((v, k) => ctx.fillText(opts.yFmt(v), m.left - 6, m.top + ph - (k * ph) / 2));
    if (opts.marker) {
      const { x, y } = opts.marker;
      const mx = m.left + ((x - x0) / (x1 - x0)) * pw;
      const my = m.top + ph - ((y - y0) / (y1 - y0)) * ph;
      if (mx >= m.left && mx <= m.left + pw && my >= m.top && my <= m.top + ph) {
        ctx.beginPath();
        ctx.arc(mx, my, 4, 0, 2 * Math.PI);
        ctx.fillStyle = P.text;
        ctx.fill();
        ctx.strokeStyle = P.bg;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }
    return { m, pw, ph };
  }

  // ---------------------------------------------------------------------------
  // Random-walk chain (Example 6.2)
  // ---------------------------------------------------------------------------

  function drawChain(ctx, w, h, opts) {
    const P = palette();
    const names = ['', 'A', 'B', 'C', 'D', 'E', ''];
    const pad = 32;
    const step = (w - 2 * pad) / 6;
    const y = h / 2;
    const r = Math.min(16, step * 0.28, h * 0.3);
    ctx.strokeStyle = P.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(w - pad, y);
    ctx.stroke();
    ctx.font = `11px ${MONO}`;
    for (let i = 0; i <= 6; i++) {
      const x = pad + i * step;
      if (i === 0 || i === 6) {
        ctx.fillStyle = P.wall;
        ctx.fillRect(x - r * 0.6, y - r * 0.6, r * 1.2, r * 1.2);
        ctx.fillStyle = P.muted;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(i === 6 ? '+1' : '0', x, y + r + 4);
        continue;
      }
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.fillStyle = P.bg;
      ctx.fill();
      ctx.strokeStyle = P.text;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = P.text;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(names[i], x, y);
    }
    if (opts.pos != null) {
      const x = pad + opts.pos * step;
      ctx.beginPath();
      ctx.arc(x, y - r - 8, 5, 0, 2 * Math.PI);
      ctx.fillStyle = P.accent;
      ctx.fill();
    }
    if (opts.caption) {
      ctx.fillStyle = P.muted;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(opts.caption, pad - 24, 4);
    }
  }

  root.Draw = { Surface, palette, withAlpha, cssVar, gridGeometry, drawGrid, drawCartPole, drawHeatmap, drawChain, fmtCell };
})(typeof globalThis !== 'undefined' ? globalThis : this);
