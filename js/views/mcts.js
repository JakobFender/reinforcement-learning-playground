// §8.11: Monte Carlo tree search as decision-time planning, plus an AlphaZero-style variant.
(function () {
  'use strict';

  const { UI, RL, Plot, Draw } = window;

  function view(container, tab) {
    const L = UI.layout(container, tab, {
      title: 'Monte Carlo tree search',
      ref: '§8.11 · §16.6',
      desc: 'Plan at decision time: before every real move, simulate many futures from the current state with a model, grow a search tree where the simulations look promising, then take the most-visited action. The AlphaZero-style variant replaces random rollouts with a value estimate and guides selection with a prior.',
      tex: [
        String.raw`\text{UCT: } a = \arg\max_a \Big[\, \bar Q(s,a) + c\sqrt{\tfrac{\ln N(s)}{N(s,a)}} \,\Big]`,
        String.raw`\text{PUCT: } a = \arg\max_a \Big[\, \bar Q(s,a) + c\, P(s,a)\tfrac{\sqrt{N(s)}}{1+N(s,a)} \,\Big]`,
      ],
      underTitle: 'Inside one <em>search</em>',
      ariaLabel: 'Gridworld with the agent, its path and where the search looked. Click to edit cells.',
    });

    const state = {
      preset: 'maze',
      gamma: 0.95,
      slip: 0,
      env: RL.GridWorld.fromPreset('maze'),
      mode: 'uct',
      sims: 500,
      c: 1.4,
      depth: 40,
      sigma: 0.1,
      seed: 1,
    };

    let rng;
    let net;
    let s;
    let trail;
    let ret;
    let moves;
    let last; // last search result
    let episodes; // lengths of finished episodes
    let geom = null;
    let vStar = null;
    let shortest = null;

    function newNet() {
      net = state.env.hasGoal() ? RL.noisyValueNet(state.env, state.gamma, state.sigma, RL.mulberry32(state.seed + 17)) : new Float64Array(state.env.nS);
    }

    function startEpisode() {
      s = state.env.startState(rng);
      trail = [s];
      ret = 0;
      moves = 0;
      last = null;
    }

    function reset() {
      rng = RL.mulberry32(state.seed);
      episodes = [];
      newNet();
      const env = state.env;
      vStar = env.hasGoal() ? RL.solveOptimal(env, state.gamma, 1e-6, 3000) : null;
      shortest = env.hasGoal() && env.starts.length ? -RL.solveOptimal(env, 1, 1e-6, 3000)[env.starts[0]] / -env.stepReward : null;
      startEpisode();
      stepper.reset();
      render();
    }

    function move() {
      const env = state.env;
      if (!env.hasGoal()) return false;
      last = RL.mctsSearch(env, s, {
        gamma: state.gamma,
        simulations: state.sims,
        c: state.mode === 'alphazero' ? state.c * 1.25 : state.c,
        rolloutDepth: state.depth,
        maxDepth: state.depth,
        mode: state.mode,
        valueNet: net,
      }, rng);
      last.s = s;
      const t = env.step(s, last.action, rng);
      ret += Math.pow(state.gamma, moves) * t.r;
      moves++;
      s = t.s2;
      trail.push(s);
      if (t.done || moves >= 300) {
        episodes.push(moves);
        last.finished = { moves, ret, reached: t.done };
        startEpisode();
        last.s = null;
      }
      return true;
    }

    // ----- Controls -----
    const c = L.controls;
    c.append(UI.heading('Environment'));
    UI.gridControls(c, state, () => reset());

    c.append(UI.heading('Search'));
    const modeSeg = UI.segmented([['uct', 'UCT with rollouts'], ['alphazero', 'AlphaZero-style']], () => state.mode, (v) => {
      state.mode = v;
      syncVisibility();
      reset();
    }, 'Search variant');
    c.append(modeSeg.el);
    const params = UI.el('div', 'params');
    params.style.marginTop = 'var(--space-3)';
    const simsS = UI.slider({ label: 'Simulations per move', min: 4, max: 5000, log: true, get: () => state.sims, set: (v) => (state.sims = Math.round(v)), change: () => render() });
    const cS = UI.slider({ label: 'Exploration c', min: 0, max: 4, step: 0.1, get: () => state.c, set: (v) => (state.c = v), change: () => render() });
    const dS = UI.slider({ label: 'Max depth', min: 5, max: 150, step: 5, get: () => state.depth, set: (v) => (state.depth = v), labelFn: (v) => `Max depth · ${v} steps per simulation`, change: () => render() });
    const sigS = UI.slider({
      label: 'Value-network error σ',
      min: 0,
      max: 1,
      step: 0.05,
      get: () => state.sigma,
      set: (v) => (state.sigma = v),
      labelFn: (v) => `Value-estimate noise σ · ${Math.round(v * 100)}% of the value range`,
      change: () => {
        newNet();
        render();
      },
    });
    params.append(simsS.el, cS.el, dS.el, sigS.el);
    c.append(params);
    const azNote = UI.el('p', 'note');
    azNote.innerHTML = 'There is no trained network here: the “network” is the true v<sub>*</sub> plus noise of size σ. Its prior is a soft-max over one-step look-ahead values.';
    c.append(azNote);
    function syncVisibility() {
      dS.el.hidden = false;
      sigS.el.hidden = state.mode !== 'alphazero';
      azNote.hidden = state.mode !== 'alphazero';
    }
    syncVisibility();

    c.append(UI.heading('Run'));
    const speed = UI.speedSlider({ min: -0.3, max: 1.7, value: 0.6, unit: 'moves' });
    c.append(speed.el);
    const run = UI.runControls({
      onStep: () => {
        move();
        render();
      },
      stepLabel: 'Move',
      onReset: () => {
        state.seed = RL.randomSeed();
        reset();
      },
    });
    c.append(run.el);
    const status = UI.el('p', 'status');
    c.append(status);

    // ----- Main canvas -----
    L.tools.append(UI.el('span', 'meta', 'Shaded: states the last search visited'));
    const surface = new Draw.Surface(L.canvas, (ctx, w, h) => {
      geom = Draw.drawGrid(ctx, w, h, state.env, {
        heat: last ? last.visits : null,
        trail,
        agent: s,
        rootWeights: last && last.s != null ? { s: last.s, w: last.counts } : null,
      });
    });
    UI.attachGridEditor(L.canvas, state, () => geom, () => reset());

    // ----- Minis -----
    const m1 = UI.mini(L.grid, 'Visits per action at the root', '', { meta: 'last move' });
    const p1 = new Plot(m1.canvas, { margin: { left: 44 } });
    const m2 = UI.mini(L.grid, 'Value of each action', '', { meta: 'Q̄(s, a)' });
    const p2 = new Plot(m2.canvas, { margin: { left: 48 } });
    m2.note.innerHTML = 'Mean simulated return through each root action. <span class="key"></span>true q<sub>*</sub>(s, a) from value iteration.';
    const m3 = UI.mini(L.grid, 'Moves per episode', '', { meta: 'shortest path dashed' });
    const p3 = new Plot(m3.canvas, { margin: { left: 44 } });
    m3.note.innerHTML = 'Each episode starts from S. MCTS keeps no memory between moves, so only the simulation budget makes it better.';

    function render() {
      surface.draw();
      const env = state.env;
      const accent = UI.cssColor('--series-1');
      const ink = UI.cssColor('--series-2');
      const muted = UI.cssColor('--series-3');
      L.setLegend([
        { label: 'search visits', color: accent, box: true },
        { label: 'path this episode', color: accent },
      ]);
      const sims = last ? last.counts.reduce((a, b) => a + b, 0) : 0;
      L.setReadout([
        { label: 'Episodes', value: String(episodes.length) },
        { label: 'Moves', value: String(moves) },
        { label: 'Discounted return', value: UI.fmt(ret, 3), accent: true },
        { label: 'Simulations', value: UI.fmtInt(state.sims) },
      ]);
      if (!env.hasGoal()) status.textContent = 'Add a goal cell so there is something to plan for.';
      else if (last && last.finished) status.innerHTML = last.finished.reached ? `<b>Reached the goal</b> in ${last.finished.moves} moves. A new episode starts from S.` : 'Gave up after 300 moves.';
      else if (last) status.innerHTML = `Chose <b>${RL.ACTIONS[last.action].name}</b> after ${sims} simulations.`;
      else status.textContent = 'Press Run or Move. Each move runs a fresh search from the agent’s state.';

      const labels = RL.ACTIONS.map((a) => a.arrow);
      if (last) {
        const tot = last.counts.reduce((a, b) => a + b, 0) || 1;
        const share = last.counts.map((n) => n / tot);
        const series = [{ kind: 'bars', ys: share, colors: share.map((_, a) => (a === last.action ? accent : muted)), color: muted, half: 0.3 }];
        if (last.prior) series.push({ kind: 'bars', ys: last.prior, color: ink, outline: true, half: 0.38 });
        p1.set({ x: { min: 0.5, max: 4.5, labels }, y: { min: 0, max: 1 }, series });
        m1.note.innerHTML = last.prior
          ? 'Filled: share of simulations through each action. Outlined: the prior P(s, a). Search sharpens a noisy prior.'
          : 'Share of simulations through each action. The most visited one is played, not the one with the best mean.';
        const sRoot = trail.length > 1 ? trail[trail.length - 2] : null;
        const qstar = sRoot != null && vStar ? RL.qFromV(env, vStar, state.gamma, sRoot) : null;
        const q = last.Q.map((v) => (isFinite(v) ? v : 0));
        const yr = UI.extent([q, qstar || [], [0]], 0.1);
        p2.set({
          x: { min: 0.5, max: 4.5, labels },
          y: { min: yr[0], max: yr[1], zero: true },
          series: [{ kind: 'bars', ys: q, colors: q.map((_, a) => (a === last.action ? accent : muted)), color: muted, half: 0.3 }].concat(
            qstar && !last.finished ? [{ kind: 'markers', xs: [1, 2, 3, 4], ys: qstar, color: ink, radius: 3.5 }] : []
          ),
        });
      } else {
        p1.set({ x: { min: 0.5, max: 4.5, labels }, y: { min: 0, max: 1 }, series: [], message: 'No search yet' });
        m1.note.textContent = 'Share of simulations through each action.';
        p2.set({ x: { min: 0.5, max: 4.5, labels }, y: { min: -1, max: 0 }, series: [], message: 'No search yet' });
      }
      const opt = shortest != null && isFinite(shortest) && shortest < 1e5 ? shortest : null;
      const ys = episodes.slice();
      const yMax = Math.max(10, ...ys, opt || 0) * 1.1;
      p3.set({
        x: { min: 0.5, max: Math.max(5, ys.length) + 0.5, integer: true },
        y: { min: 0, max: yMax },
        series: [
          { kind: 'line', xs: [0.5, Math.max(5, ys.length) + 0.5], ys: [opt, opt], color: ink, dash: [4, 4], width: 1, alpha: opt ? 0.7 : 0 },
          { kind: 'line', xs: ys.map((_, i) => i + 1), ys, color: accent, width: 2 },
          { kind: 'markers', xs: ys.map((_, i) => i + 1), ys, color: accent, radius: 3 },
        ],
        message: ys.length ? null : 'No finished episode yet',
      });
    }

    const stepper = UI.makeStepper();
    reset();

    return {
      frame(dt) {
        if (!run.running) return;
        // Searches are expensive: at most a few per frame.
        const n = stepper.advance(dt, speed.rate(), move, 30);
        if (n) render();
      },
      redraw: render,
      retex: L.retex,
      destroy() {},
    };
  }

  window.Views.mcts = view;
})();
