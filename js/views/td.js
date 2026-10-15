// §6.1–6.2: TD(0) prediction vs constant-α Monte Carlo on the five-state random walk (Example 6.2).
(function () {
  'use strict';

  const { UI, RL, Plot, Draw } = window;

  const NAMES = ['A', 'B', 'C', 'D', 'E'];

  function view(container, tab) {
    const L = UI.layout(container, tab, {
      title: 'TD(0) prediction',
      ref: '§6.1–6.2 · Example 6.2',
      desc: 'Estimate the values of a fixed policy, one step at a time. TD(0) updates each state towards the reward plus the current estimate of the next state, so it learns before the episode is over. Here it races constant-α Monte Carlo on a random walk whose true values are 1/6 … 5/6.',
      tex: [
        String.raw`\text{TD(0): } V(S_t) \leftarrow V(S_t) + \alpha\,\big[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\big]`,
        String.raw`\text{MC: } V(S_t) \leftarrow V(S_t) + \alpha\,\big[G_t - V(S_t)\big]`,
      ],
      strip: true,
      underTitle: 'Which one learns <em>faster</em>?',
      minis: 2,
      ariaLabel: 'Estimated state values for states A to E',
    });

    const state = { alphaTD: 0.1, alphaMC: 0.02, runs: 100, seed: 1 };
    let rng;
    let Vtd;
    let Vmc;
    let episodes;
    let rmsTD;
    let rmsMC;
    let snapshots; // TD estimates after 0, 1, 10, 100 episodes (Figure 6.2, left)
    let ep; // current episode { states, i }
    let pos;
    let lastReward = null;

    function reset() {
      rng = RL.mulberry32(state.seed);
      Vtd = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0];
      Vmc = Vtd.slice();
      episodes = 0;
      rmsTD = [RL.rmsError(Vtd)];
      rmsMC = [RL.rmsError(Vmc)];
      snapshots = { 0: Vtd.slice() };
      ep = null;
      pos = 3;
      lastReward = null;
      stepper.reset();
      render();
    }

    // One transition of the walk. TD updates now; MC waits for the end of the episode.
    function step() {
      if (!ep) ep = { states: [3], i: 0 };
      const s = ep.states[ep.states.length - 1];
      const s2 = s + (rng() < 0.5 ? -1 : 1);
      ep.states.push(s2);
      const r = s2 === 6 ? 1 : 0;
      const v2 = s2 === 0 || s2 === 6 ? 0 : Vtd[s2];
      Vtd[s] += state.alphaTD * (r + v2 - Vtd[s]);
      pos = s2;
      if (s2 === 0 || s2 === 6) {
        RL.randomWalkUpdate(Vmc, { states: ep.states, reward: r }, 'mc', state.alphaMC);
        episodes++;
        lastReward = r;
        rmsTD.push(RL.rmsError(Vtd));
        rmsMC.push(RL.rmsError(Vmc));
        if (episodes === 1 || episodes === 10 || episodes === 100) snapshots[episodes] = Vtd.slice();
        ep = null;
        return true;
      }
      return false;
    }

    function runEpisode() {
      while (!step());
    }

    // ----- Controls -----
    const c = L.controls;
    c.append(UI.heading('Step sizes'));
    const ps = UI.el('div', 'params');
    ps.append(
      UI.slider({ label: 'TD(0) step size α', min: 0.01, max: 0.5, step: 0.01, get: () => state.alphaTD, set: (v) => (state.alphaTD = v), change: () => { curves = null; render(); } }).el,
      UI.slider({ label: 'Monte Carlo step size α', min: 0.01, max: 0.5, step: 0.01, get: () => state.alphaMC, set: (v) => (state.alphaMC = v), change: () => { curves = null; render(); } }).el
    );
    c.append(ps);
    const note = UI.el('p', 'note');
    note.innerHTML = 'Policy: move left or right with equal probability. Reward +1 only for leaving on the right, γ = 1. Both start at V = 0.5.';
    c.append(note);

    c.append(UI.heading('Run'));
    const speed = UI.speedSlider({ min: 0, max: 3.5, value: 1, unit: 'steps' });
    c.append(speed.el);
    const run = UI.runControls({
      onStep: () => {
        step();
        render();
      },
      extra: [
        { label: 'Episode', fn: () => { runEpisode(); render(); } },
        { label: '+100', title: 'Run 100 episodes', fn: () => { for (let i = 0; i < 100; i++) runEpisode(); render(); } },
      ],
      onReset: () => {
        state.seed = RL.randomSeed();
        reset();
      },
    });
    c.append(run.el);
    const status = UI.el('p', 'status');
    c.append(status);

    // ----- Main charts -----
    const strip = new Draw.Surface(L.strip, (ctx, w, h) => Draw.drawChain(ctx, w, h, { pos }));
    const plot = new Plot(L.canvas, {
      margin: { left: 48 },
      onHover: (h) => {
        const i = Math.round(h.x);
        if (i < 1 || i > 5) return null;
        return [`<b>state</b> ${NAMES[i - 1]}`, `<b>true</b> ${UI.fmt(RL.RANDOM_WALK_TRUE[i], 3)}`, `<b>TD(0)</b> ${UI.fmt(Vtd[i], 3)}`, `<b>MC</b> ${UI.fmt(Vmc[i], 3)}`];
      },
    });

    // ----- Minis -----
    const m1 = UI.mini(L.grid, 'Average RMS error over many runs', '', { meta: 'Figure 6.2, right' });
    const p1 = new Plot(m1.canvas, {
      margin: { left: 48 },
      onHover: (h) => {
        if (!curves) return null;
        const e = Math.round(h.x);
        if (e < 0 || e > 100) return null;
        return [`<b>episode</b> ${e}`].concat(curves.map((cv) => `<b>${cv.label}</b> ${UI.fmt(cv.ys[e], 3)}`));
      },
    });
    m1.note.innerHTML = 'RMS error over the five states, averaged over 100 runs. <span class="key accent"></span>TD(0) <span class="key muted"></span>MC. Bold lines use your step sizes, thin ones the book’s.';
    const m2 = UI.mini(L.grid, 'This run', '', { meta: 'RMS error per episode' });
    const p2 = new Plot(m2.canvas, { margin: { left: 48 } });
    m2.note.innerHTML = 'The same error for the single run above. <span class="key accent"></span>TD(0) <span class="key muted"></span>MC';

    let curves = null;
    function computeCurves() {
      const list = [];
      for (const a of [0.05, 0.15]) list.push({ label: `TD α=${a}`, method: 'td', a, mine: false });
      for (const a of [0.01, 0.03, 0.04]) list.push({ label: `MC α=${a}`, method: 'mc', a, mine: false });
      list.push({ label: `TD α=${state.alphaTD}`, method: 'td', a: state.alphaTD, mine: true });
      list.push({ label: `MC α=${state.alphaMC}`, method: 'mc', a: state.alphaMC, mine: true });
      curves = list.map((cv) => Object.assign(cv, { ys: RL.randomWalkCurve(cv.method, cv.a, 100, state.runs, 3) }));
    }

    function render() {
      strip.draw();
      const accent = UI.cssColor('--series-1');
      const ink = UI.cssColor('--series-2');
      const muted = UI.cssColor('--series-3');
      const xs = [1, 2, 3, 4, 5];
      const series = [{ kind: 'line', xs, ys: RL.RANDOM_WALK_TRUE.slice(1, 6), color: ink, dash: [5, 4], width: 1.5 }];
      for (const k of [0, 1, 10, 100]) {
        if (!snapshots[k]) continue;
        series.push({ kind: 'line', xs, ys: snapshots[k].slice(1, 6), color: muted, width: 1, alpha: 0.5 });
      }
      series.push({ kind: 'line', xs, ys: Vmc.slice(1, 6), color: muted, width: 2 });
      series.push({ kind: 'points', xs, ys: Vmc.slice(1, 6), color: muted, radius: 3.5 });
      series.push({ kind: 'line', xs, ys: Vtd.slice(1, 6), color: accent, width: 2 });
      series.push({ kind: 'points', xs, ys: Vtd.slice(1, 6), color: accent, radius: 4 });
      plot.set({ x: { min: 0.6, max: 5.4, labels: NAMES }, y: { min: 0, max: 1 }, series });
      L.setLegend([
        { label: 'true', color: ink, dashed: true },
        { label: 'TD(0)', color: accent },
        { label: 'MC', color: muted },
        { label: 'TD after 0 · 1 · 10 · 100', color: muted },
      ]);
      L.setReadout([
        { label: 'Episodes', value: UI.fmtInt(episodes) },
        { label: 'RMS error · TD(0)', value: UI.fmt(RL.rmsError(Vtd), 3), accent: true },
        { label: 'RMS error · MC', value: UI.fmt(RL.rmsError(Vmc), 3) },
      ]);
      if (ep) status.innerHTML = `Walking: ${ep.states.map((s) => (s === 0 ? '◁' : s === 6 ? '▷' : NAMES[s - 1])).join(' ')}`;
      else if (lastReward != null) status.innerHTML = `Episode ${episodes} ended on the <b>${lastReward ? 'right (+1)' : 'left (0)'}</b>. MC updated every state it visited at once.`;
      else status.textContent = 'Press Run to watch the walk, or Episode to play one at a time.';

      if (!curves) computeCurves();
      const ser = [];
      for (const cv of curves) {
        ser.push({ kind: 'line', xs: Array.from(cv.ys, (_, i) => i), ys: Array.from(cv.ys), color: cv.method === 'td' ? accent : muted, width: cv.mine ? 2.25 : 1, alpha: cv.mine ? 1 : 0.6 });
      }
      p1.set({ x: { min: 0, max: 100, integer: true }, y: { min: 0, max: 0.25 }, series: ser });

      const ex = rmsTD.map((_, i) => i);
      p2.set({
        x: { min: 0, max: Math.max(10, episodes), integer: true },
        y: { min: 0, max: Math.max(0.25, ...rmsTD, ...rmsMC) * 1.05 },
        series: [
          { kind: 'line', xs: ex, ys: rmsMC, color: muted, width: 1.75 },
          { kind: 'line', xs: ex, ys: rmsTD, color: accent, width: 2 },
        ],
      });
    }

    const stepper = UI.makeStepper();
    reset();

    return {
      frame(dt) {
        if (!run.running) return;
        const n = stepper.advance(dt, speed.rate(), () => { step(); }, 10);
        if (n) render();
      },
      redraw: render,
      retex: L.retex,
      destroy() {},
    };
  }

  window.Views.td = view;
})();
