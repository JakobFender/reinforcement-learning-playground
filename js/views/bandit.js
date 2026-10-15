// Chapter 2: the 10-armed testbed with ε-greedy, optimistic initial values, UCB and the gradient bandit.
(function () {
  'use strict';

  const { UI, RL, Plot } = window;

  const INFO = {
    egreedy: {
      title: 'ε-greedy action selection',
      ref: '§2.2–2.4',
      desc: 'Act greedily on your value estimates most of the time, but with probability ε pick an arm at random so every arm keeps getting sampled.',
      tex: [
        String.raw`A_t = \begin{cases} \arg\max_a Q_t(a) & \text{prob. } 1-\varepsilon \\ \text{random arm} & \text{prob. } \varepsilon \end{cases}`,
        String.raw`Q_{n+1} = Q_n + \alpha\,\big[R_n - Q_n\big], \quad \alpha = \tfrac{1}{n} \text{ or constant}`,
      ],
      compare: 'greedy',
      study: { key: 'epsilon', label: 'ε', min: 1 / 128, max: 1 / 2, base: {} },
    },
    optimistic: {
      title: 'Optimistic initial values',
      ref: '§2.6',
      desc: 'Start every estimate far above any real reward. Each pull disappoints, so even a purely greedy agent tries all arms early on.',
      tex: [
        String.raw`Q_1(a) = Q_{\text{init}} \gg q_*(a), \qquad A_t = \arg\max_a Q_t(a)`,
        String.raw`Q_{n+1} = Q_n + \alpha\,\big[R_n - Q_n\big]`,
      ],
      compare: 'realistic',
      study: { key: 'q0', label: 'Q₁', min: 1 / 4, max: 8, base: {} },
    },
    ucb: {
      title: 'Upper confidence bound',
      ref: '§2.7',
      desc: 'Prefer arms whose value is uncertain: the bonus grows with time and shrinks every time an arm is pulled.',
      tex: [String.raw`A_t = \arg\max_a \Big[\, Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \,\Big]`],
      compare: 'eps',
      study: { key: 'c', label: 'c', min: 1 / 16, max: 8, base: {} },
    },
    gradient: {
      title: 'Gradient bandit',
      ref: '§2.8',
      desc: 'Learn a preference per arm instead of a value, and pick arms with a soft-max. The average reward R̄ acts as a baseline.',
      tex: [
        String.raw`\pi_t(a) = \frac{e^{H_t(a)}}{\sum_b e^{H_t(b)}}`,
        String.raw`H_{t+1}(a) = H_t(a) + \alpha\,(R_t - \bar R_t)\big(\mathbb{1}_{a=A_t} - \pi_t(a)\big)`,
      ],
      compare: 'nobaseline',
      study: { key: 'alpha', label: 'α', min: 1 / 32, max: 4, base: {} },
    },
  };

  const COMPARE = {
    none: { label: 'Nothing' },
    greedy: { label: 'Greedy (ε = 0)', method: 'egreedy', p: { epsilon: 0, alpha: 0, q0: 0 } },
    eps: { label: 'ε-greedy (ε = 0.1)', method: 'egreedy', p: { epsilon: 0.1, alpha: 0, q0: 0 } },
    realistic: { label: 'Realistic ε-greedy (Q₁ = 0, ε = 0.1, α = 0.1)', method: 'egreedy', p: { epsilon: 0.1, alpha: 0.1, q0: 0 } },
    optimistic: { label: 'Optimistic greedy (Q₁ = 5, α = 0.1)', method: 'optimistic', p: { epsilon: 0, alpha: 0.1, q0: 5 } },
    ucb: { label: 'UCB (c = 2)', method: 'ucb', p: { c: 2, alpha: 0, q0: 0 } },
    gradient: { label: 'Gradient bandit (α = 0.1)', method: 'gradient', p: { alpha: 0.1, baseline: true } },
    nobaseline: { label: 'Gradient bandit without baseline', method: 'gradient', p: { alpha: 0.1, baseline: false } },
  };

  // Figure 2.6: average reward over the first 1000 steps as a function of each method's parameter.
  const STUDY = [
    { method: 'egreedy', key: 'epsilon', name: 'ε-greedy', exps: [-7, -6, -5, -4, -3, -2], base: { alpha: 0, q0: 0 } },
    { method: 'gradient', key: 'alpha', name: 'gradient', exps: [-5, -4, -3, -2, -1, 0, 1, 2], base: { baseline: true } },
    { method: 'ucb', key: 'c', name: 'UCB', exps: [-4, -3, -2, -1, 0, 1, 2], base: { alpha: 0, q0: 0 } },
    { method: 'optimistic', key: 'q0', name: 'optimistic', exps: [-2, -1, 0, 1, 2], base: { epsilon: 0, alpha: 0.1 } },
  ];

  function view(container, tab) {
    const method = tab.method;
    const info = INFO[method];
    const L = UI.layout(container, tab, Object.assign({}, info, {
      underTitle: 'One bandit, <em>up close</em>',
      ariaLabel: 'Learning curve averaged over many runs',
    }));

    const state = {
      p: Object.assign({}, RL.banditMethods[method].defaults),
      k: 10,
      runs: 500,
      steps: 1000,
      nonstationary: false,
      offset: method === 'gradient' ? 4 : 0,
      compare: info.compare,
      view: 'reward',
      seed: 1,
    };

    // ----- Controls -----
    const c = L.controls;
    c.append(UI.heading('Parameters'));
    const params = UI.el('div', 'params');
    const sliders = [];
    const changed = () => restart();
    if (method === 'egreedy' || method === 'optimistic') {
      sliders.push(UI.slider({ label: 'Exploration ε', min: 0, max: 0.5, step: 0.01, get: () => state.p.epsilon, set: (v) => (state.p.epsilon = v), change: changed }));
    }
    if (method === 'ucb') {
      sliders.push(UI.slider({ label: 'Confidence c', min: 0, max: 5, step: 0.05, get: () => state.p.c, set: (v) => (state.p.c = v), change: changed }));
    }
    if (method !== 'gradient') {
      sliders.push(UI.slider({
        label: 'Initial estimate Q₁',
        min: -2,
        max: 10,
        step: 0.5,
        get: () => state.p.q0,
        set: (v) => (state.p.q0 = v),
        change: changed,
      }));
      sliders.push(UI.slider({
        label: 'Step size α',
        min: 0,
        max: 1,
        step: 0.01,
        get: () => state.p.alpha,
        set: (v) => (state.p.alpha = v),
        labelFn: (v) => (v === 0 ? 'Step size α · 0 means 1/n (sample average)' : 'Step size α · constant'),
        change: changed,
      }));
    } else {
      sliders.push(UI.slider({ label: 'Step size α', min: 0.01, max: 4, log: true, get: () => state.p.alpha, set: (v) => (state.p.alpha = v), change: changed }));
    }
    sliders.forEach((s) => params.append(s.el));
    c.append(params);
    if (method === 'gradient') {
      const cb = UI.checkbox('Use the average reward as baseline', () => state.p.baseline, (v) => {
        state.p.baseline = v;
        restart();
      });
      cb.el.style.marginTop = 'var(--space-3)';
      c.append(cb.el);
    }

    c.append(UI.heading('Testbed'));
    const tb = UI.el('div', 'params');
    tb.append(
      UI.slider({ label: 'Arms k', min: 2, max: 20, step: 1, get: () => state.k, set: (v) => (state.k = v), change: changed }).el,
      UI.slider({
        label: 'Shift every q*(a) by',
        min: 0,
        max: 5,
        step: 0.5,
        get: () => state.offset,
        set: (v) => (state.offset = v),
        labelFn: (v) => (v ? `Shift every q*(a) by +${v} (Figure 2.5 uses +4)` : 'Shift every q*(a) by'),
        change: changed,
      }).el
    );
    c.append(tb);
    const ns = UI.checkbox('Non-stationary: q*(a) take a random walk', () => state.nonstationary, (v) => {
      state.nonstationary = v;
      restart();
    });
    ns.el.style.marginTop = 'var(--space-3)';
    c.append(ns.el);

    c.append(UI.heading('Experiment'));
    const ex = UI.el('div', 'params');
    ex.append(
      UI.select('Independent runs', [[100, '100'], [500, '500'], [2000, '2,000 (as in the book)']], () => state.runs, (v) => {
        state.runs = Number(v);
        restart();
      }).el,
      UI.select('Steps per run', [[1000, '1,000'], [5000, '5,000'], [10000, '10,000']], () => state.steps, (v) => {
        state.steps = Number(v);
        restart();
      }).el,
      UI.select('Compare with', Object.entries(COMPARE).map(([k, v]) => [k, v.label]), () => state.compare, (v) => {
        state.compare = v;
        restart();
      }).el
    );
    c.append(ex);
    const btns = UI.el('div', 'row-btns');
    btns.append(UI.button('New random testbeds', 'btn ghost', () => {
      state.seed = RL.randomSeed();
      restart();
    }));
    c.append(btns);
    const status = UI.el('p', 'status');
    c.append(status);

    // ----- Main chart -----
    const toggle = UI.segmented([['reward', 'Average reward'], ['optimal', '% optimal action']], () => state.view, (v) => {
      state.view = v;
      render();
    }, 'Metric');
    L.tools.append(toggle.el);
    const plot = new Plot(L.canvas, {
      margin: { left: 56 },
      onHover: (h) => {
        const t = Math.round(h.x);
        if (!res || t < 1 || t > res.cfg.steps) return null;
        const get = (r) => (state.view === 'reward' ? UI.fmt(r.reward[t - 1] / Math.max(1, r.done), 3) : `${UI.fmt((100 * r.optimal[t - 1]) / Math.max(1, r.done), 3)}%`);
        const lines = [`<b>step</b> ${t}`, `<b>${RL.banditMethods[method].name}</b> ${get(res.main)}`];
        if (res.cmp) lines.push(`<b>compare</b> ${get(res.cmp)}`);
        return lines;
      },
    });

    // ----- Minis -----
    const m1 = UI.mini(L.grid, method === 'gradient' ? 'Preferences vs truth' : 'Estimates vs truth', '', { meta: 'last run' });
    const p1 = new Plot(m1.canvas, {
      margin: { left: 40 },
      onHover: (h) => {
        const a = Math.round(h.x) - 1;
        if (!res || !res.main.last || a < 0 || a >= res.cfg.k) return null;
        const l = res.main.last;
        const lines = [`<b>arm</b> ${a + 1}`, `<b>q*</b> ${UI.fmt(l.q[a])}`];
        if (method === 'gradient') lines.push(`<b>H</b> ${UI.fmt(l.H[a])}`, `<b>π</b> ${UI.fmt(l.pi[a])}`);
        else lines.push(`<b>Q</b> ${UI.fmt(l.Q[a])}`);
        lines.push(`<b>pulls</b> ${l.N[a]}`);
        return lines;
      },
    });
    const m2 = UI.mini(L.grid, 'Pulls per arm', 'Bars count how often each arm was chosen in the last run. The best arm is in olive.', { meta: 'last run' });
    const p2 = new Plot(m2.canvas, { margin: { left: 48 } });
    const m3 = UI.mini(L.grid, 'Parameter study', '', { meta: 'Figure 2.6' });
    const p3 = new Plot(m3.canvas, {
      margin: { left: 40 },
      onHover: (h) => {
        if (!study.points.length) return null;
        const lines = [`<b>parameter</b> ${UI.fmt(h.x, 2)}`];
        for (const s of STUDY) {
          const pts = study.points.filter((p) => p.method === s.method);
          if (!pts.length) continue;
          let best = pts[0];
          for (const p of pts) if (Math.abs(Math.log(p.x / h.x)) < Math.abs(Math.log(best.x / h.x))) best = p;
          lines.push(`<b>${s.name}</b> ${UI.fmt(best.y, 3)}`);
        }
        return lines;
      },
    });
    m1.note.innerHTML =
      method === 'gradient'
        ? 'Outlined bars are the true values q*(a), filled bars the learned preferences H(a). Only their order matters.'
        : 'Outlined bars are the true values q*(a), filled bars the agent’s estimates Q(a) after the last run.';
    m3.note.innerHTML = 'Average reward over the first 1,000 steps for a range of settings (200 runs each). Click to use a value. <span class="key accent"></span>this method';
    const studyBtn = UI.button('Run parameter study', 'btn ghost', () => startStudy());
    const sb = UI.el('div', 'row-btns');
    sb.append(studyBtn);
    m3.fig.append(sb);
    m3.canvas.style.cursor = 'pointer';
    m3.canvas.addEventListener('click', (e) => {
      if (!p3.spec) return;
      const { px, py } = p3.localPoint(e);
      if (!p3.inPlot(px, py)) return;
      const { x } = p3.pxToData(px, py);
      const key = info.study.key;
      const s = STUDY.find((q) => q.method === method);
      const lo = Math.pow(2, s.exps[0]);
      const hi = Math.pow(2, s.exps[s.exps.length - 1]);
      state.p[key] = Number(UI.clamp(x, lo, hi).toPrecision(2));
      if (method === 'optimistic' && key === 'q0') state.p.q0 = Math.round(state.p.q0 * 2) / 2;
      sliders.forEach((sl) => sl.sync());
      restart();
    });

    // ----- Simulation (progressive, a few runs per frame) -----
    let res = null;

    function makeRunner(m, p, cfg) {
      const rng = RL.mulberry32(cfg.seed);
      return {
        m,
        p,
        rng,
        reward: new Float64Array(cfg.steps),
        optimal: new Float64Array(cfg.steps),
        done: 0,
        last: null,
        oneRun() {
          const bandit = RL.makeBandit(cfg.k, rng, { nonstationary: cfg.nonstationary, offset: cfg.offset });
          const agent = RL.createBanditAgent(m, p, cfg.k, rng);
          for (let t = 0; t < cfg.steps; t++) {
            const best = bandit.optimal();
            const a = agent.act();
            const r = bandit.pull(a);
            agent.update(a, r);
            this.reward[t] += r;
            if (a === best) this.optimal[t]++;
          }
          this.done++;
          this.last = { q: bandit.q.slice(), Q: Array.from(agent.Q), N: Array.from(agent.N), H: Array.from(agent.H), pi: agent.pi.slice(), best: bandit.optimal() };
        },
      };
    }

    // Experiments are computed into `pending` over several frames; the charts only switch to a result
    // once every run is finished, so they never show a half-averaged, jittery curve.
    let pending = null;

    function restart() {
      const cfg = { steps: state.steps, k: state.k, runs: state.runs, nonstationary: state.nonstationary, offset: state.offset, compare: state.compare, seed: state.seed, p: Object.assign({}, state.p) };
      const cmp = COMPARE[cfg.compare];
      pending = {
        cfg,
        main: makeRunner(method, cfg.p, cfg),
        cmp: cmp.method ? makeRunner(cmp.method, Object.assign({}, cmp.p), cfg) : null,
      };
      renderStatus();
    }

    const finished = (r) => r.main.done >= r.cfg.runs && (!r.cmp || r.cmp.done >= r.cfg.runs);

    function frame() {
      if (pending) {
        const t0 = performance.now();
        while (!finished(pending) && performance.now() - t0 < 14) {
          if (pending.main.done < pending.cfg.runs) pending.main.oneRun();
          if (pending.cmp && pending.cmp.done < pending.cfg.runs) pending.cmp.oneRun();
        }
        if (finished(pending)) {
          res = pending;
          pending = null;
          render();
        } else renderStatus();
      }
      if (study.running) {
        stepStudy();
        if (study.running) m3.meta.textContent = `computing ${study.total - study.queue.length} / ${study.total}`;
        else render();
      }
    }

    // ----- Parameter study -----
    const study = { running: false, queue: [], points: [], next: [] };
    function startStudy() {
      study.queue = [];
      study.next = [];
      for (const s of STUDY) for (const e of s.exps) study.queue.push({ s, x: Math.pow(2, e) });
      study.total = study.queue.length;
      study.running = true;
      studyBtn.disabled = true;
    }
    function stepStudy() {
      const t0 = performance.now();
      while (study.queue.length && performance.now() - t0 < 10) {
        const { s, x } = study.queue.shift();
        const p = Object.assign({}, RL.banditMethods[s.method].defaults, s.base, { [s.key]: x });
        const r = RL.runBandits(s.method, p, { runs: 200, steps: 1000, k: state.k, seed: 7, offset: state.offset, nonstationary: state.nonstationary });
        let sum = 0;
        for (const v of r.avgReward) sum += v;
        study.next.push({ method: s.method, x, y: sum / r.avgReward.length });
      }
      if (!study.queue.length) {
        study.points = study.next;
        study.running = false;
        studyBtn.disabled = false;
        studyBtn.textContent = 'Run again';
      }
    }

    // ----- Rendering -----
    function renderStatus() {
      if (pending) {
        const r = pending;
        const total = r.cfg.runs * (r.cmp ? 2 : 1);
        const done = r.main.done + (r.cmp ? r.cmp.done : 0);
        status.innerHTML = `Simulating <b>${UI.fmtInt(r.main.done)} / ${UI.fmtInt(r.cfg.runs)}</b> runs… ${Math.round((100 * done) / total)}%`;
      } else if (res) {
        status.innerHTML = `Each curve averages ${UI.fmtInt(res.cfg.runs)} runs, each on a freshly drawn bandit with q*(a) ~ N(${res.cfg.offset}, 1).`;
      }
      if (!res) plot.set({ x: { min: 1, max: state.steps, integer: true }, y: { min: 0, max: 1.5 }, series: [], message: 'Simulating…' });
    }

    function render() {
      renderStatus();
      if (!res) return;
      const cfg = res.cfg;
      const accent = UI.cssColor('--series-1');
      const ink = UI.cssColor('--series-2');
      const muted = UI.cssColor('--series-3');
      const metric = (r) => {
        const src = state.view === 'reward' ? r.reward : r.optimal;
        const n = Math.max(1, r.done);
        const ys = new Float64Array(src.length);
        for (let i = 0; i < src.length; i++) ys[i] = (state.view === 'reward' ? src[i] : 100 * src[i]) / n;
        return ys;
      };
      const series = [];
      const all = [];
      if (res.cmp && res.cmp.done) {
        const ys = metric(res.cmp);
        all.push(ys);
        const t = UI.thin(ys, 800);
        series.push({ kind: 'line', xs: t.xs, ys: t.ys, color: ink, width: 1.5, dash: [5, 4], alpha: 0.75 });
      }
      if (res.main.done) {
        const ys = metric(res.main);
        all.push(ys);
        const t = UI.thin(ys, 800);
        series.push({ kind: 'line', xs: t.xs, ys: t.ys, color: accent, width: 2 });
      }
      let yr;
      if (state.view === 'optimal') yr = [0, 100];
      else {
        yr = UI.extent(all.length ? all : [[0, 1.5]], 0.08);
        yr[0] = Math.min(yr[0], cfg.offset);
      }
      plot.set({ x: { min: 1, max: cfg.steps, integer: true }, y: { min: yr[0], max: yr[1], zero: true }, series, crosshair: true });
      const legend = [{ label: RL.banditMethods[method].name, color: accent }];
      if (res.cmp) legend.push({ label: COMPARE[cfg.compare].label, color: ink, dashed: true });
      L.setLegend(legend);

      // Readout: averages over the final 10% of steps.
      const tail = Math.max(1, Math.floor(cfg.steps / 10));
      const avgTail = (r, src, mult) => {
        if (!r || !r.done) return NaN;
        let s = 0;
        for (let i = cfg.steps - tail; i < cfg.steps; i++) s += src[i];
        return (mult * s) / tail / r.done;
      };
      const items = [
        { label: 'Runs', value: `${UI.fmtInt(res.main.done)}` },
        { label: `Reward · last ${UI.fmtInt(tail)} steps`, value: UI.fmt(avgTail(res.main, res.main.reward, 1), 3), accent: true },
        { label: '% optimal', value: `${UI.fmt(avgTail(res.main, res.main.optimal, 100), 3)}%` },
      ];
      if (res.cmp) items.push({ label: 'Compare · reward', value: UI.fmt(avgTail(res.cmp, res.cmp.reward, 1), 3) });
      L.setReadout(items);

      // Mini 1: estimates vs truth.
      const l = res.main.last;
      if (l) {
        const est = method === 'gradient' ? l.H : l.Q;
        const yr1 = UI.extent([l.q, est, [0]], 0.1);
        const labels = l.q.map((_, i) => String(i + 1));
        p1.set({
          x: { min: 0.5, max: cfg.k + 0.5, labels },
          y: { min: yr1[0], max: yr1[1], zero: true },
          series: [
            { kind: 'bars', ys: est, color: accent, half: 0.22 },
            { kind: 'bars', ys: l.q, color: ink, outline: true, half: 0.36 },
          ],
        });
        const colors = l.N.map((_, i) => (i === l.best ? accent : muted));
        p2.set({
          x: { min: 0.5, max: cfg.k + 0.5, labels },
          y: { min: 0, max: Math.max(...l.N) * 1.08 || 1 },
          series: [{ kind: 'bars', ys: l.N, colors, color: muted, half: 0.3 }],
        });
      }

      // Mini 3: parameter study.
      const ss = [];
      for (const s of STUDY) {
        const pts = study.points.filter((p) => p.method === s.method);
        if (!pts.length) continue;
        const mine = s.method === method;
        ss.push({ kind: 'line', xs: pts.map((p) => p.x), ys: pts.map((p) => p.y), color: mine ? accent : muted, width: mine ? 2 : 1.25, alpha: mine ? 1 : 0.8 });
        ss.push({ kind: 'markers', xs: pts.map((p) => p.x), ys: pts.map((p) => p.y), color: mine ? accent : muted, radius: mine ? 3 : 2 });
      }
      const ys3 = study.points.map((p) => p.y);
      const yr3 = ys3.length ? UI.extent([ys3], 0.1) : [cfg.offset + 1, cfg.offset + 1.6];
      const curX = state.p[info.study.key];
      const vlines = curX > 0 ? [{ x: curX, color: ink, dash: [3, 3], label: `${info.study.label} = ${UI.fmt(curX, 2)}` }] : [];
      p3.set({
        x: { min: Math.log10(1 / 128), max: Math.log10(8), log: true },
        y: { min: yr3[0], max: yr3[1] },
        series: ss,
        vlines,
        message: study.points.length ? null : study.running ? 'Computing…' : 'Press “Run parameter study”',
      });
      if (!study.running) m3.meta.textContent = 'Figure 2.6';
    }

    restart();

    return {
      frame,
      redraw: render,
      retex: L.retex,
      destroy() {},
    };
  }

  window.Views.bandit = view;
})();
