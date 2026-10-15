// Chapters 5–6: Monte Carlo control, Sarsa, Expected Sarsa and Q-learning on an editable gridworld.
(function () {
  'use strict';

  const { UI, RL, Plot, Draw } = window;

  const NAMES = { mc: 'Monte Carlo', sarsa: 'Sarsa', expected: 'Expected Sarsa', qlearning: 'Q-learning' };

  const INFO = {
    mc: {
      title: 'Monte Carlo control',
      ref: '§5.4',
      desc: 'Play a whole episode with an ε-soft policy, then move every visited Q(s, a) towards the return that actually followed its first visit. No bootstrapping and no model: values come only from complete experience.',
      tex: [
        String.raw`G_t = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{T-t-1} R_T`,
        String.raw`Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\,\big[G_t - Q(S_t,A_t)\big]`,
      ],
      preset: 'maze',
      alpha: 0,
      compare: 'qlearning',
    },
    sarsa: {
      title: 'Sarsa',
      ref: '§6.4',
      desc: 'On-policy TD control. After every step, update towards the reward plus the value of the action the ε-greedy policy actually takes next: S, A, R, S′, A′.',
      tex: [String.raw`Q(S,A) \leftarrow Q(S,A) + \alpha\,\big[R + \gamma\, Q(S',A') - Q(S,A)\big]`],
      preset: 'windy',
      alpha: 0.5,
      compare: 'qlearning',
    },
    expected: {
      title: 'Expected Sarsa',
      ref: '§6.6',
      desc: 'Like Sarsa, but instead of sampling the next action, average over all of them under the current policy. Same expected update, less variance, so larger step sizes work.',
      tex: [String.raw`Q(S,A) \leftarrow Q(S,A) + \alpha\,\Big[R + \gamma \sum_a \pi(a\mid S')\, Q(S',a) - Q(S,A)\Big]`],
      preset: 'cliff',
      alpha: 0.9,
      compare: 'sarsa',
    },
    qlearning: {
      title: 'Q-learning',
      ref: '§6.5',
      desc: 'Off-policy TD control. Behave ε-greedily, but learn about the greedy policy by bootstrapping from the best next action. On the cliff it learns the optimal edge path, and keeps falling off while exploring.',
      tex: [String.raw`Q(S,A) \leftarrow Q(S,A) + \alpha\,\big[R + \gamma \max_a Q(S',a) - Q(S,A)\big]`],
      preset: 'cliff',
      alpha: 0.5,
      compare: 'sarsa',
    },
  };

  function view(container, tab) {
    const method = tab.method;
    const info = INFO[method];
    const L = UI.layout(container, tab, Object.assign({}, info, {
      underTitle: 'How it <em>learns</em>',
      ariaLabel: 'Gridworld with action values, greedy policy and the agent. Click to edit cells.',
    }));

    const state = {
      preset: info.preset,
      gamma: RL.gridPresets[info.preset].gamma,
      slip: 0,
      env: RL.GridWorld.fromPreset(info.preset),
      alpha: info.alpha,
      epsilon: 0.1,
      compare: info.compare,
      show: 'q',
      seed: 1,
    };

    let agent;
    let shadow;
    let hover = null;
    let geom = null;
    let stepsAtEpisode; // cumulative steps when each episode ended (Figure 6.3)
    let shadowSteps;

    const params = () => ({ alpha: state.alpha, epsilon: state.epsilon, gamma: state.gamma, maxSteps: method === 'mc' ? 2000 : 5000 });

    function reset() {
      agent = RL.createTabularAgent(state.env, method, params(), RL.mulberry32(state.seed));
      shadow = state.compare !== 'none' ? RL.createTabularAgent(state.env, state.compare, Object.assign(params(), { maxSteps: state.compare === 'mc' ? 2000 : 5000, alpha: state.compare === 'mc' ? 0 : state.alpha }), RL.mulberry32(state.seed + 1)) : null;
      stepsAtEpisode = [];
      shadowSteps = [];
      stepper.reset();
      render();
    }

    // Parameters change in place: the learner continues with the new values.
    function updateParams() {
      Object.assign(agent.params, params());
      if (shadow) Object.assign(shadow.params, { epsilon: state.epsilon, gamma: state.gamma, alpha: state.compare === 'mc' ? 0 : state.alpha });
      render();
    }

    function step() {
      const info2 = agent.step();
      if (info2.done) {
        stepsAtEpisode.push(agent.totalSteps);
        // Keep the comparison in lockstep by episodes.
        if (shadow) {
          while (shadow.episodes < agent.episodes) {
            shadow.runEpisode(20000);
            shadowSteps.push(shadow.totalSteps);
          }
        }
      }
      return info2;
    }

    function runEpisode() {
      const target = agent.episodes + 1;
      let n = 0;
      while (agent.episodes < target && n++ < 20000) step();
    }

    // ----- Controls -----
    const c = L.controls;
    c.append(UI.heading('Environment'));
    UI.gridControls(c, state, (kind) => {
      if (kind === 'gamma' || kind === 'slip') updateParams();
      else reset();
    });

    c.append(UI.heading('Parameters'));
    const ps = UI.el('div', 'params');
    if (method === 'mc') {
      ps.append(UI.slider({
        label: 'Step size α',
        min: 0,
        max: 1,
        step: 0.01,
        get: () => state.alpha,
        set: (v) => (state.alpha = v),
        labelFn: (v) => (v === 0 ? 'Step size α · 0 means 1/n (average of returns)' : 'Step size α · constant'),
        change: updateParams,
      }).el);
    } else {
      ps.append(UI.slider({ label: 'Step size α', min: 0.01, max: 1, step: 0.01, get: () => state.alpha, set: (v) => (state.alpha = v), change: updateParams }).el);
    }
    ps.append(UI.slider({
      label: 'Exploration ε',
      min: 0,
      max: 0.5,
      step: 0.01,
      get: () => state.epsilon,
      set: (v) => (state.epsilon = v),
      labelFn: (v) => `Exploration ε${v === 0 ? ' · greedy (may never finish an episode)' : ''}`,
      change: updateParams,
    }).el);
    c.append(ps);
    const cmpOpts = [['none', 'Nothing']].concat(Object.keys(NAMES).filter((k) => k !== method).map((k) => [k, NAMES[k]]));
    const cmp = UI.select('Compare with', cmpOpts, () => state.compare, (v) => {
      state.compare = v;
      reset();
    });
    cmp.el.style.marginTop = 'var(--space-3)';
    c.append(cmp.el);
    const cmpNote = UI.el('p', 'note', 'The comparison learner trains on the same gridworld for the same number of episodes, in the background.');
    c.append(cmpNote);

    c.append(UI.heading('Run'));
    const speed = UI.speedSlider({ min: 0, max: 4.5, value: 1.8, unit: 'steps' });
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

    // ----- Main canvas -----
    const showSeg = UI.segmented([['q', 'Q(s, a)'], ['v', 'max Q + policy'], ['visits', 'Visits']], () => state.show, (v) => {
      state.show = v;
      render();
    }, 'Show');
    L.tools.append(showSeg.el);
    const surface = new Draw.Surface(L.canvas, (ctx, w, h) => {
      const env = state.env;
      const o = { hover, agent: agent.s != null ? agent.s : null, trail: agent.s != null ? agent.trajectory : agent.lastEpisode };
      if (state.show === 'q') {
        o.qs = agent.Q;
        o.mask = agent.visits;
        o.policy = greedy();
      } else if (state.show === 'v') {
        const v = agent.Q.map((q) => Math.max(...q));
        o.values = v;
        o.numbers = v;
        o.mask = agent.visits;
        o.policy = greedy();
      } else {
        o.heat = agent.visits;
        o.path = agent.episodes ? agent.greedyPath(env.nS) : null;
      }
      geom = Draw.drawGrid(ctx, w, h, env, o);
    });
    function greedy() {
      return agent.Q.map((q, s) => {
        if (!agent.visits[s]) return null;
        const best = RL.argmaxAll(q, 1e-9);
        if (best.length === 4) return null;
        const p = [0, 0, 0, 0];
        best.forEach((a) => (p[a] = 1));
        return p;
      });
    }
    UI.attachGridEditor(L.canvas, state, () => geom, () => reset(), (s) => {
      hover = s;
      if (!run.running) render();
    });

    // ----- Minis -----
    const m1 = UI.mini(L.grid, 'Return per episode', '', { meta: 'Figure 6.4' });
    const p1 = new Plot(m1.canvas, {
      margin: { left: 52 },
      onHover: (h) => {
        const i = Math.round(h.x);
        if (i < 1 || i > agent.returns.length) return null;
        const lines = [`<b>episode</b> ${i}`, `<b>${NAMES[method]}</b> ${UI.fmt(agent.returns[i - 1], 4)}`];
        if (shadow && shadow.returns[i - 1] != null) lines.push(`<b>${NAMES[state.compare]}</b> ${UI.fmt(shadow.returns[i - 1], 4)}`);
        return lines;
      },
    });
    m1.note.innerHTML = 'Undiscounted sum of rewards in each episode, smoothed over 10. <span class="key accent"></span>this method <span class="key dashed"></span>comparison';
    const m2 = UI.mini(L.grid, 'Episodes vs time steps', '', { meta: 'Figure 6.3' });
    const p2 = new Plot(m2.canvas, { margin: { left: 44 } });
    m2.note.textContent = 'Episodes completed as a function of total steps. The slope gets steeper as episodes get shorter.';
    const m3 = UI.mini(L.grid, 'Action values', '', { meta: 'hover a cell' });
    const p3 = new Plot(m3.canvas, { margin: { left: 52 } });
    m3.note.innerHTML = 'Q(s, a) for the hovered state, or the start. The greedy action is olive.';

    function render() {
      surface.draw();
      const env = state.env;
      const accent = UI.cssColor('--series-1');
      const ink = UI.cssColor('--series-2');
      const muted = UI.cssColor('--series-3');
      L.setLegend(
        state.show === 'visits'
          ? [{ label: 'visits (log)', color: accent, box: true }, { label: 'greedy path', color: ink, dashed: true }]
          : [{ label: 'higher value', color: accent, box: true }, { label: 'greedy action', color: ink }]
      );
      const n = agent.returns.length;
      const recent = agent.returns.slice(-20);
      const avg = recent.length ? recent.reduce((a, b) => a + b, 0) / recent.length : NaN;
      let pathLen = '–';
      if (agent.episodes) {
        const path = agent.greedyPath(env.nS);
        pathLen = env.isTerminal(path[path.length - 1]) ? String(path.length - 1) : 'no goal';
      }
      L.setReadout([
        { label: 'Episodes', value: UI.fmtInt(agent.episodes) },
        { label: 'Steps', value: UI.fmtInt(agent.totalSteps) },
        { label: 'Return · last 20', value: UI.fmt(avg, 3), accent: true },
        { label: 'Greedy path', value: pathLen, title: 'Length of the path the greedy policy takes from the start' },
      ]);
      if (!env.hasGoal()) status.textContent = 'There is no goal: episodes only end at the step limit.';
      else if (agent.s != null) status.innerHTML = `Episode ${agent.episodes + 1} · step ${agent.epSteps}`;
      else if (agent.lastTruncated) status.innerHTML = `Episode ${agent.episodes} hit the step limit before reaching the goal.`;
      else if (n) status.innerHTML = `Episode ${n} took <b>${agent.lengths[n - 1]}</b> steps, return ${UI.fmt(agent.returns[n - 1], 4)}.`;
      else status.textContent = 'All Q start at zero. Press Run, or Step to go one move at a time.';

      // Mini 1: returns per episode
      const series = [];
      const all = [];
      if (shadow && shadow.returns.length) {
        const sm = UI.smooth(shadow.returns, 10);
        all.push(sm);
        const t = UI.thin(sm, 600);
        series.push({ kind: 'line', xs: t.xs, ys: t.ys, color: ink, dash: [5, 4], width: 1.5, alpha: 0.75 });
      }
      if (n) {
        const sm = UI.smooth(agent.returns, 10);
        all.push(sm);
        const t = UI.thin(sm, 600);
        series.push({ kind: 'line', xs: t.xs, ys: t.ys, color: accent, width: 2 });
      }
      const yr = UI.extent(all.length ? all : [[-100, 0]], 0.08);
      p1.set({ x: { min: 1, max: Math.max(10, n), integer: true }, y: { min: Math.max(yr[0], -2000), max: Math.max(yr[1], 0) + 1 }, series, message: n ? null : 'No episodes yet' });

      // Mini 2: episodes vs steps
      const s2 = [];
      if (shadow && shadowSteps.length) {
        const t = UI.thin(shadowSteps.map((_, i) => i + 1), 600, shadowSteps);
        s2.push({ kind: 'line', xs: [0].concat(t.xs), ys: [0].concat(t.ys), color: ink, dash: [5, 4], width: 1.5, alpha: 0.75 });
      }
      if (stepsAtEpisode.length) {
        const t = UI.thin(stepsAtEpisode.map((_, i) => i + 1), 600, stepsAtEpisode);
        s2.push({ kind: 'line', xs: [0].concat(t.xs), ys: [0].concat(t.ys), color: accent, width: 2 });
      }
      const xMax = Math.max(100, agent.totalSteps, shadowSteps.length ? shadowSteps[shadowSteps.length - 1] : 0);
      p2.set({ x: { min: 0, max: xMax }, y: { min: 0, max: Math.max(5, stepsAtEpisode.length) * 1.05 }, series: s2 });

      // Mini 3: Q at a state
      let s0 = hover != null && env.isState(hover) && !env.isTerminal(hover) ? hover : env.startState(RL.mulberry32(1));
      if (s0 == null) return;
      const q = Array.from(agent.Q[s0]);
      const best = RL.argmaxAll(q, 1e-9);
      const yr3 = UI.extent([q, [0]], 0.12);
      p3.set({
        x: { min: 0.5, max: 4.5, labels: RL.ACTIONS.map((a) => a.arrow) },
        y: { min: yr3[0], max: yr3[1], zero: true },
        series: [{ kind: 'bars', ys: q, colors: q.map((_, a) => (best.length < 4 && best.includes(a) ? accent : muted)), color: muted, half: 0.3 }],
      });
      m3.meta.textContent = `row ${Math.floor(s0 / env.cols) + 1}, col ${(s0 % env.cols) + 1}`;
    }

    const stepper = UI.makeStepper();
    reset();

    return {
      frame(dt) {
        if (!run.running) return;
        const n = stepper.advance(dt, speed.rate(), () => { step(); }, 14);
        if (n) render();
      },
      redraw: render,
      retex: L.retex,
      destroy() {},
    };
  }

  window.Views.tabular = view;
})();
