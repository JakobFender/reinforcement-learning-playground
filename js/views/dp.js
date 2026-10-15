// Chapter 4: iterative policy evaluation, policy iteration and value iteration on an editable gridworld.
(function () {
  'use strict';

  const { UI, RL, Plot, Draw } = window;

  const INFO = {
    evaluation: {
      title: 'Iterative policy evaluation',
      ref: '§4.1',
      desc: 'Compute v<sub>π</sub> for a fixed policy, here the equiprobable random one, by sweeping the Bellman expectation equation over every state until nothing changes. Arrows show the greedy policy with respect to the current estimate, as in Figure 4.1.',
      tex: [String.raw`v_{k+1}(s) = \sum_a \pi(a\mid s) \sum_{s',r} p(s',r\mid s,a)\big[r + \gamma\, v_k(s')\big]`],
      preset: 'small',
    },
    policy: {
      title: 'Policy iteration',
      ref: '§4.3',
      desc: 'Alternate two steps: evaluate the current policy, then make it greedy with respect to its own value function. A finite MDP has finitely many policies, so this stops at an optimal one, usually after very few iterations.',
      tex: [
        String.raw`\pi_0 \xrightarrow{E} v_{\pi_0} \xrightarrow{I} \pi_1 \xrightarrow{E} v_{\pi_1} \xrightarrow{I} \cdots \xrightarrow{I} \pi_* \xrightarrow{E} v_*`,
        String.raw`\pi'(s) = \arg\max_a \sum_{s',r} p(s',r\mid s,a)\big[r + \gamma\, v_\pi(s')\big]`,
      ],
      preset: 'rooms',
    },
    value: {
      title: 'Value iteration',
      ref: '§4.4',
      desc: 'Fold improvement into every sweep: back up the best action instead of the policy’s average. Values spread outward from the goal one step per sweep.',
      tex: [String.raw`v_{k+1}(s) = \max_a \sum_{s',r} p(s',r\mid s,a)\big[r + \gamma\, v_k(s')\big]`],
      preset: 'maze',
    },
  };

  function view(container, tab) {
    const method = tab.method;
    const info = INFO[method];
    const L = UI.layout(container, tab, Object.assign({}, info, {
      underTitle: method === 'policy' ? 'How fast it <em>settles</em>' : 'How fast it <em>converges</em>',
      minis: 2,
      ariaLabel: 'Gridworld showing state values and policy arrows. Click to edit cells.',
    }));

    const env0 = RL.GridWorld.fromPreset(info.preset);
    const state = {
      preset: info.preset,
      gamma: RL.gridPresets[info.preset].gamma,
      slip: 0,
      env: env0,
      theta: 1e-4,
      inPlace: true,
      maxEval: method === 'policy' ? 1000 : 0,
      show: 'values',
    };

    let V;
    let policy;
    let sweeps;
    let deltas;
    let improvements; // sweep index at which each improvement happened
    let changes; // states changed per improvement
    let phase; // 'evaluate' | 'done' | 'diverged'
    let evalSweeps;
    let hover = null;
    let selected = null;
    let geom = null;

    function reset() {
      V = new Float64Array(state.env.nS);
      policy = RL.uniformPolicy(state.env);
      sweeps = 0;
      deltas = [];
      improvements = [];
      changes = [];
      phase = 'evaluate';
      evalSweeps = 0;
      stepper.reset();
      render();
    }

    // One unit of work: a sweep (plus an improvement step for policy iteration when evaluation converged).
    function step() {
      if (phase !== 'evaluate') return false;
      const env = state.env;
      let d;
      if (method === 'value') d = RL.valueIterationSweep(env, V, state.gamma, state.inPlace);
      else d = RL.evaluationSweep(env, policy, V, state.gamma, state.inPlace);
      sweeps++;
      evalSweeps++;
      deltas.push(d);
      let big = 0;
      for (let s = 0; s < env.nS; s++) big = Math.max(big, Math.abs(V[s]));
      if (!isFinite(big) || big > 1e7) {
        phase = 'diverged';
        return false;
      }
      if (method === 'policy') {
        const limit = state.maxEval >= 100 ? Infinity : state.maxEval;
        if (d < state.theta || evalSweeps >= limit) {
          const next = RL.greedyPolicy(env, V, state.gamma);
          let n = 0;
          for (let s = 0; s < env.nS; s++) {
            if (!env.isState(s) || env.isTerminal(s)) continue;
            const a = RL.argmaxAll(next[s])[0];
            if (!(policy[s][a] > 0) || RL.argmaxAll(policy[s]).length !== RL.argmaxAll(next[s]).length) n++;
          }
          const stable = RL.samePolicy(next, policy);
          policy = next;
          improvements.push(sweeps);
          changes.push(n);
          evalSweeps = 0;
          // Stable policy and converged values: done (with truncated evaluation, wait for Δ < θ too).
          if (stable && d < state.theta) phase = 'done';
        }
      } else if (d < state.theta) phase = 'done';
      return phase === 'evaluate';
    }

    // ----- Controls -----
    const c = L.controls;
    c.append(UI.heading('Environment'));
    UI.gridControls(c, state, (kind) => {
      if (kind === 'env') hover = selected = null;
      reset();
    });

    c.append(UI.heading('Algorithm'));
    const params = UI.el('div', 'params');
    params.append(
      UI.select('Stop when Δ <', [[1e-2, 'θ = 0.01'], [1e-4, 'θ = 0.0001'], [1e-6, 'θ = 0.000001']], () => state.theta, (v) => {
        state.theta = Number(v);
        if (phase === 'done') phase = 'evaluate';
        render();
      }).el
    );
    if (method === 'policy') {
      params.append(
        UI.slider({
          label: 'Evaluation sweeps per iteration',
          min: 1,
          max: 100,
          step: 1,
          get: () => state.maxEval >= 100 ? 100 : state.maxEval,
          set: (v) => (state.maxEval = v >= 100 ? 1000 : v),
          labelFn: (v) => (v >= 100 ? 'Evaluation sweeps per iteration · until Δ < θ' : `Evaluation sweeps per iteration · at most ${v} (modified policy iteration)`),
          change: reset,
        }).el
      );
    }
    c.append(params);
    const ip = UI.checkbox('In-place updates (use new values within a sweep)', () => state.inPlace, (v) => {
      state.inPlace = v;
      reset();
    });
    ip.el.style.marginTop = 'var(--space-3)';
    c.append(ip.el);

    c.append(UI.heading('Run'));
    const speed = UI.speedSlider({ min: -0.3, max: 2, value: method === 'evaluation' ? 0.5 : 0.7, unit: 'sweeps' });
    c.append(speed.el);
    const run = UI.runControls({
      onStep: () => {
        if (phase === 'done' || phase === 'diverged') return;
        step();
        render();
      },
      stepLabel: 'Sweep',
      extra: [{
        label: 'Finish',
        fn: () => {
          const t0 = performance.now();
          while (step() && performance.now() - t0 < 1500);
          render();
        },
      }],
      onReset: reset,
    });
    c.append(run.el);
    const status = UI.el('p', 'status');
    c.append(status);

    // ----- Main canvas -----
    const showToggle = UI.segmented([['values', 'Values + policy'], ['values-only', 'Values'], ['policy', 'Policy']], () => state.show, (v) => {
      state.show = v;
      render();
    }, 'Show');
    L.tools.append(showToggle.el);
    const surface = new Draw.Surface(L.canvas, (ctx, w, h) => {
      const showV = state.show !== 'policy';
      const showP = state.show !== 'values-only';
      const arrows = method === 'policy' ? policy : RL.greedyPolicy(state.env, V, state.gamma);
      geom = Draw.drawGrid(ctx, w, h, state.env, {
        values: showV ? V : null,
        numbers: showV ? V : null,
        policy: showP ? arrows : null,
        hover,
        selected,
      });
    });
    UI.attachGridEditor(L.canvas, state, () => geom, () => reset(), (s) => {
      if (s === hover) return;
      hover = s;
      surface.draw();
    }, (s) => {
      selected = s === selected ? null : s;
      render();
    });

    // ----- Minis -----
    const m1 = UI.mini(L.grid, 'Largest change per sweep', '', { meta: 'Δ, log scale' });
    const p1 = new Plot(m1.canvas, {
      margin: { left: 48 },
      onHover: (h) => {
        const i = Math.round(h.x);
        if (i < 1 || i > deltas.length) return null;
        return [`<b>sweep</b> ${i}`, `<b>Δ</b> ${UI.fmt(deltas[i - 1], 3)}`];
      },
    });
    m1.note.innerHTML =
      method === 'policy'
        ? 'Each spike is a policy improvement: the new policy has different values, so evaluation starts moving again. Dotted lines mark improvements.'
        : `Sweeps stop once Δ falls below θ. With γ &lt; 1 the error shrinks at least by a factor γ per sweep${method === 'value' ? '' : ' (a contraction)'}.`;
    const m2 = UI.mini(L.grid, 'One Bellman backup', '', { meta: 'click a cell' });
    const p2 = new Plot(m2.canvas, { margin: { left: 48 } });
    m2.note.innerHTML = 'For the selected state (or the start): the value of each action, r + γ v(s′) averaged over outcomes. <span class="key"></span>v(s)';

    function backupState() {
      const env = state.env;
      if (selected != null && env.isState(selected) && !env.isTerminal(selected)) return selected;
      if (env.starts.length) return env.starts[0];
      for (let s = 0; s < env.nS; s++) if (env.isState(s) && !env.isTerminal(s)) return s;
      return null;
    }

    function render() {
      surface.draw();
      const accent = UI.cssColor('--series-1');
      const ink = UI.cssColor('--series-2');
      const muted = UI.cssColor('--series-3');
      const env = state.env;

      // Legend + readout
      L.setLegend([
        { label: 'higher value', color: accent, box: true },
        { label: method === 'policy' ? 'policy π' : 'greedy w.r.t. v', color: ink },
      ]);
      const s0 = backupState();
      const items = [
        { label: 'Sweeps', value: UI.fmtInt(sweeps) },
        { label: 'Δ', value: deltas.length ? UI.fmt(deltas[deltas.length - 1], 2) : '–' },
      ];
      if (method === 'policy') items.push({ label: 'Improvements', value: String(improvements.length) });
      if (s0 != null) items.push({ label: selected != null && selected === s0 ? 'v(selected)' : env.starts.length ? 'v(start)' : 'v(s)', value: UI.fmt(V[s0], 4), accent: true });
      L.setReadout(items);

      if (phase === 'diverged') {
        status.innerHTML = '<b>Values diverge.</b> Some states never reach the goal under this policy, so with γ = 1 their value is −∞. Lower γ or add a goal.';
        run.set(false);
      } else if (!env.hasGoal() && state.gamma === 1) {
        status.innerHTML = 'There is no goal and γ = 1, so values will diverge. Add a goal cell or lower γ.';
      } else if (phase === 'done') {
        status.innerHTML =
          method === 'policy'
            ? `<b>Policy is stable</b> after ${improvements.length} improvement${improvements.length === 1 ? '' : 's'} and ${sweeps} sweeps: it is optimal.`
            : `<b>Converged</b> after ${sweeps} sweeps (Δ &lt; ${state.theta}).`;
        run.set(false);
      } else if (method === 'policy') {
        status.innerHTML = `Evaluating π<sub>${improvements.length}</sub> · sweep ${evalSweeps}${changes.length ? ` · last improvement changed ${changes[changes.length - 1]} state${changes[changes.length - 1] === 1 ? '' : 's'}` : ''}`;
      } else status.textContent = sweeps ? 'Sweeping…' : 'All values start at zero. Press Run, or Sweep to go one step at a time.';

      // Mini 1: Δ per sweep
      const ys = deltas.map((d) => Math.max(d, 1e-12));
      const lo = ys.length ? Math.min(...ys, state.theta) : state.theta;
      const hi = ys.length ? Math.max(...ys) : 1;
      const xMax = Math.max(10, deltas.length + 0.5);
      const xs = ys.map((_, i) => i + 1);
      p1.set({
        x: { min: 0.5, max: xMax, integer: true },
        y: { min: Math.floor(Math.log10(lo)), max: Math.ceil(Math.log10(hi) + 0.01), log: true },
        series: ys.length
          ? [
              { kind: 'line', xs: [0.5, xMax], ys: [state.theta, state.theta], color: ink, width: 1, dash: [4, 4], alpha: 0.6 },
              { kind: 'line', xs, ys, color: accent, width: 2 },
              { kind: 'markers', xs, ys, color: accent, radius: ys.length > 60 ? 0 : 2.5 },
            ]
          : [],
        vlines: improvements.map((x) => ({ x: x + 0.5, color: muted, dash: [2, 3] })),
        message: ys.length ? null : 'No sweeps yet',
      });

      // Mini 2: q-values at a state
      if (s0 != null) {
        const q = RL.qFromV(env, V, state.gamma, s0);
        const yr = UI.extent([q, [V[s0], 0]], 0.12);
        const best = RL.argmaxAll(q, 1e-9);
        p2.set({
          x: { min: 0.5, max: 4.5, labels: RL.ACTIONS.map((a) => a.arrow) },
          y: { min: yr[0], max: yr[1], zero: true },
          series: [
            { kind: 'bars', ys: q, colors: q.map((_, a) => (best.includes(a) ? accent : muted)), color: muted, half: 0.3 },
            { kind: 'line', xs: [0.5, 4.5], ys: [V[s0], V[s0]], color: ink, width: 1.5 },
          ],
        });
        const r = Math.floor(s0 / env.cols);
        m2.meta.textContent = `row ${r + 1}, col ${(s0 % env.cols) + 1}`;
      }
    }

    const stepper = UI.makeStepper();
    reset();

    return {
      frame(dt) {
        if (!run.running) return;
        const n = stepper.advance(dt, speed.rate(), step, 20);
        if (n || phase !== 'evaluate') render();
      },
      redraw: render,
      retex: L.retex,
      destroy() {},
    };
  }

  window.Views.dp = view;
})();
