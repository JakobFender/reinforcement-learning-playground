// Function approximation on cart-pole: deep Q-learning, REINFORCE and actor–critic with small neural networks.
(function () {
  'use strict';

  const { UI, RL, Plot, Draw } = window;

  const INFO = {
    dqn: {
      title: 'Deep Q-learning',
      ref: '§16.5 · §10.1 · §11.3',
      desc: 'Q-learning with a neural network instead of a table. Two tricks from the DQN paper tame the “deadly triad” of function approximation, bootstrapping and off-policy learning: an experience replay buffer and a slowly moving target network.',
      tex: [
        String.raw`\mathcal{L}(\theta) = \Big(R + \gamma \max_{a'} \hat q(S',a';\theta^-) - \hat q(S,A;\theta)\Big)^2`,
        String.raw`\theta^- \leftarrow (1-\tau)\,\theta^- + \tau\,\theta`,
      ],
      make: RL.createDQN,
      defaults: RL.dqnDefaults,
    },
    reinforce: {
      title: 'REINFORCE',
      ref: '§13.3–13.4',
      desc: 'Learn the policy directly. After each episode, make every action more likely in proportion to how much better its return was than a learned baseline v̂(s). Unbiased, but noisy: it waits for complete returns.',
      tex: [String.raw`\theta \leftarrow \theta + \alpha\, \gamma^t \big(G_t - \hat v(S_t,\mathbf{w})\big) \nabla_\theta \ln \pi(A_t \mid S_t, \theta)`],
      make: RL.createReinforce,
      defaults: RL.reinforceDefaults,
    },
    ac: {
      title: 'Actor–critic',
      ref: '§13.5 · Chapter 7',
      desc: 'A policy (the actor) and a value function (the critic) learn together. The critic’s TD error δ tells the actor whether an action turned out better or worse than expected, so it can learn online, before the episode ends. Here δ uses n-step returns.',
      tex: [
        String.raw`\delta = R_{t+1} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n \hat v(S_{t+n},\mathbf{w}) - \hat v(S_t,\mathbf{w})`,
        String.raw`\mathbf{w} \leftarrow \mathbf{w} + \alpha^{\mathbf{w}} \delta\, \nabla \hat v(S_t,\mathbf{w}), \qquad \theta \leftarrow \theta + \alpha^{\theta} \delta\, \nabla \ln \pi(A_t\mid S_t,\theta)`,
      ],
      make: RL.createActorCritic,
      defaults: RL.acDefaults,
    },
  };

  const MAP = { nx: 36, ny: 28, theta: [-0.21, 0.21], omega: [-2, 2] };

  function view(container, tab) {
    const method = tab.method;
    const info = INFO[method];
    const L = UI.layout(container, tab, Object.assign({}, info, {
      underTitle: 'What the network <em>thinks</em>',
      ariaLabel: 'Cart-pole simulation',
    }));

    const state = { p: Object.assign({}, info.defaults), watch: 'train', seed: 1 };
    let agent;
    let demo; // greedy demo episode
    let lastAction = null;
    let maps = null;
    let mapsAt = 0;

    function reset() {
      agent = info.make(Object.assign({}, state.p), RL.mulberry32(state.seed));
      demo = null;
      maps = null;
      mapsAt = 0;
      trainStepper.reset();
      render(true);
    }

    // Live parameter changes: the agents read their parameters on every update.
    function live() {
      Object.assign(agent.p, state.p);
      render();
    }

    // ----- Controls -----
    const c = L.controls;
    const envNote = UI.el('p', 'note');
    envNote.innerHTML =
      method === 'dqn'
        ? 'Cart-pole (Example 3.4): push left or right to keep the pole within 12° and the cart on the track, for up to 500 steps. DQN uses the book’s alternative reward, −1 on failure and 0 otherwise, which keeps Q between −1 and 0.'
        : 'Cart-pole (Example 3.4): push left or right to keep the pole within 12° and the cart on the track, for up to 500 steps. Reward +1 per step.';
    c.append(UI.heading('Environment'), envNote);

    c.append(UI.heading('Parameters'));
    const ps = UI.el('div', 'params');
    const p = state.p;
    const sl = (o) => ps.append(UI.slider(Object.assign({ change: live }, o)).el);
    if (method === 'dqn') {
      sl({ label: 'Learning rate', min: 1e-4, max: 1e-2, log: true, get: () => p.lr, set: (v) => (p.lr = v) });
      sl({ label: 'Discount γ', min: 0.8, max: 0.999, step: 0.001, get: () => p.gamma, set: (v) => (p.gamma = v) });
      sl({ label: 'ε decay', min: 500, max: 50000, log: true, get: () => p.epsDecay, set: (v) => (p.epsDecay = Math.round(v)), labelFn: (v) => `ε decays from 1 to ${p.epsEnd} over ${UI.fmtInt(v)} steps` });
      sl({ label: 'Target update τ', min: 0.001, max: 1, log: true, get: () => p.tau, set: (v) => (p.tau = v), labelFn: (v) => `Target network update τ${v >= 1 ? ' · copy every step' : ''}` });
      sl({ label: 'Batch size', min: 1, max: 128, step: 1, get: () => p.batch, set: (v) => (p.batch = v) });
    } else if (method === 'reinforce') {
      sl({ label: 'Policy step size α', min: 1e-4, max: 0.05, log: true, get: () => p.lr, set: (v) => (p.lr = v) });
      sl({ label: 'Baseline step size α_w', min: 1e-4, max: 0.05, log: true, get: () => p.lrValue, set: (v) => (p.lrValue = v) });
      sl({ label: 'Discount γ', min: 0.8, max: 0.999, step: 0.001, get: () => p.gamma, set: (v) => (p.gamma = v) });
    } else {
      sl({ label: 'n-step return', min: 1, max: 64, step: 1, get: () => p.n, set: (v) => (p.n = v), labelFn: (v) => (v === 1 ? 'n-step return · n = 1, the one-step actor–critic' : `n-step return · n = ${v}`) });
      sl({ label: 'Actor step size α_θ', min: 1e-4, max: 0.02, log: true, get: () => p.lrActor, set: (v) => (p.lrActor = v) });
      sl({ label: 'Critic step size α_w', min: 1e-4, max: 0.05, log: true, get: () => p.lrCritic, set: (v) => (p.lrCritic = v) });
      sl({ label: 'Discount γ', min: 0.8, max: 0.999, step: 0.001, get: () => p.gamma, set: (v) => (p.gamma = v) });
    }
    c.append(ps);
    const checks = UI.el('div', 'checks');
    checks.style.marginTop = 'var(--space-3)';
    const cb = (label, key) => checks.append(UI.checkbox(label, () => p[key], (v) => { p[key] = v; live(); }).el);
    if (method === 'dqn') {
      cb('Experience replay', 'useReplay');
      cb('Target network', 'useTarget');
      cb('Double DQN (§6.7)', 'double');
    } else if (method === 'reinforce') {
      cb('Learned baseline v̂(s)', 'baseline');
      cb('Include the γᵗ factor', 'discountFactor');
    }
    if (checks.childNodes.length) c.append(checks);
    const hid = UI.select('Hidden units per layer (restarts)', [[16, '16'], [32, '32'], [64, '64']], () => p.hidden, (v) => {
      p.hidden = Number(v);
      reset();
    });
    hid.el.style.marginTop = 'var(--space-3)';
    c.append(hid.el);
    const arch = UI.el('p', 'note');
    arch.textContent =
      method === 'dqn'
        ? 'Network: 4 inputs → two ReLU layers → 2 action values. Adam optimizer, Huber loss.'
        : 'Networks: 4 inputs → one ReLU layer → 2 action preferences (soft-max) and → 1 state value. Adam optimizer.';
    c.append(arch);

    c.append(UI.heading('Run'));
    const speed = UI.speedSlider({ min: 0, max: 5, value: 3.5, unit: 'steps' });
    c.append(speed.el);
    const run = UI.runControls({
      onReset: () => {
        state.seed = RL.randomSeed();
        reset();
      },
      extra: [{ label: '+10k steps', title: 'Train for 10,000 steps', fn: () => burst(10000) }],
    });
    c.append(run.el);
    const status = UI.el('p', 'status');
    c.append(status);

    function burst(n) {
      const target = agent.steps + n;
      const t0 = performance.now();
      while (agent.steps < target && performance.now() - t0 < 4000) agent.step();
      render(true);
    }

    // ----- Main canvas -----
    const watch = UI.segmented([['train', 'Training episode'], ['greedy', 'Greedy policy']], () => state.watch, (v) => {
      state.watch = v;
      demo = null;
      render();
    }, 'Watch');
    L.tools.append(watch.el);
    const surface = new Draw.Surface(L.canvas, (ctx, w, h) => {
      if (state.watch === 'greedy') {
        const st = demo ? demo.env.state : [0, 0, 0, 0];
        Draw.drawCartPole(ctx, w, h, st, { action: demo ? demo.action : null, caption: [`greedy policy · step ${demo ? demo.env.t : 0}`, demo && demo.best != null ? `last demo lasted ${demo.best} steps` : 'real-time, 50 steps per second'] });
      } else {
        Draw.drawCartPole(ctx, w, h, agent.state, { action: lastAction, caption: [`training episode ${agent.episodes + 1} · step ${agent.epLen}`, method === 'dqn' ? `ε = ${UI.fmt(agent.epsilon(), 2)}` : 'actions sampled from π'] });
      }
    });

    // ----- Minis -----
    const m1 = UI.mini(L.grid, 'Steps balanced per episode', '', { meta: '500 = solved' });
    const p1 = new Plot(m1.canvas, {
      margin: { left: 44 },
      onHover: (h) => {
        const i = Math.round(h.x);
        if (i < 1 || i > agent.lengths.length) return null;
        return [`<b>episode</b> ${i}`, `<b>length</b> ${agent.lengths[i - 1]}`];
      },
    });
    m1.note.innerHTML = 'Each dot is a training episode; the line averages the last 20. Exploration keeps training episodes shorter than the greedy policy.';
    const m2 = UI.mini(L.grid, method === 'dqn' ? 'Greedy action' : 'Policy π(right | s)', '', { meta: 'pole angle × angular velocity' });
    const c2 = m2.canvas;
    const s2 = new Draw.Surface(c2, (ctx, w, h) => drawMap(ctx, w, h, 'policy'));
    m2.note.innerHTML = 'Cart centred and still; x-axis pole angle (rad), y-axis angular velocity (rad/s). <span class="key accent"></span>push right <span class="key muted"></span>push left. The dot is the training state.';
    const m3 = UI.mini(L.grid, method === 'dqn' ? 'State value max Q' : method === 'ac' ? 'Critic v̂(s)' : 'Baseline v̂(s)', '', { meta: 'same slice' });
    const s3 = new Draw.Surface(m3.canvas, (ctx, w, h) => drawMap(ctx, w, h, 'value'));
    m3.note.innerHTML = 'Darker olive means the network expects to keep the pole up for longer. Leaning and falling further is worth less.';

    function computeMaps() {
      const { nx, ny } = MAP;
      const pol = new Float64Array(nx * ny);
      const val = new Float64Array(nx * ny);
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const th = MAP.theta[0] + ((i + 0.5) / nx) * (MAP.theta[1] - MAP.theta[0]);
          const om = MAP.omega[0] + ((j + 0.5) / ny) * (MAP.omega[1] - MAP.omega[0]);
          const st = [0, 0, th, om];
          if (method === 'dqn') {
            const q = agent.q(st);
            pol[j * nx + i] = q[1] > q[0] ? 1 : 0;
            val[j * nx + i] = Math.max(q[0], q[1]);
          } else {
            pol[j * nx + i] = agent.prob(st);
            val[j * nx + i] = agent.value(st);
          }
        }
      }
      let lo = Infinity;
      let hi = -Infinity;
      for (const v of val) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      maps = { pol, val, lo, hi };
    }

    function drawMap(ctx, w, h, which) {
      if (!maps) return;
      const P = Draw.palette();
      const fmtA = (v) => UI.fmt(v, 2);
      const marker = state.watch === 'train' ? { x: agent.state[2], y: agent.state[3] } : demo ? { x: demo.env.state[2], y: demo.env.state[3] } : null;
      if (which === 'policy') {
        Draw.drawHeatmap(ctx, w, h, { nx: MAP.nx, ny: MAP.ny, values: maps.pol }, {
          colorOf: (v) => (v >= 0.5 ? Draw.withAlpha(P.accent, 0.1 + 1.6 * (v - 0.5)) : Draw.withAlpha(P.muted, 0.1 + 1.2 * (0.5 - v))),
          xRange: MAP.theta,
          yRange: MAP.omega,
          xFmt: fmtA,
          yFmt: fmtA,
          marker,
        });
      } else {
        const { lo, hi } = maps;
        const unused = method === 'reinforce' && !p.baseline;
        Draw.drawHeatmap(ctx, w, h, { nx: MAP.nx, ny: MAP.ny, values: maps.val }, {
          colorOf: (v) => (unused ? P.surface : Draw.withAlpha(P.accent, 0.05 + 0.8 * (hi > lo ? (v - lo) / (hi - lo) : 0))),
          xRange: MAP.theta,
          yRange: MAP.omega,
          xFmt: fmtA,
          yFmt: fmtA,
          marker,
        });
      }
    }

    function render(full) {
      const now = performance.now();
      if (full || !maps || now - mapsAt > 400) {
        computeMaps();
        mapsAt = now;
        s2.draw();
        s3.draw();
        renderStats();
      } else {
        s2.draw();
        s3.draw();
      }
      surface.draw();
    }

    function renderStats() {
      const accent = UI.cssColor('--series-1');
      const ink = UI.cssColor('--series-2');
      const muted = UI.cssColor('--series-3');
      const lens = agent.lengths;
      const n = lens.length;
      const recent = lens.slice(-20);
      const avg = recent.length ? recent.reduce((a, b) => a + b, 0) / recent.length : NaN;
      const items = [
        { label: 'Episodes', value: UI.fmtInt(agent.episodes) },
        { label: 'Steps', value: UI.fmtInt(agent.steps) },
        { label: 'Avg length · last 20', value: UI.fmt(avg, 3), accent: true },
      ];
      if (method === 'dqn') items.push({ label: 'ε', value: UI.fmt(agent.epsilon(), 2) });
      L.setReadout(items);
      L.setLegend([{ label: 'pole', color: accent }, { label: 'push', color: ink }]);
      m3.note.innerHTML =
        method === 'reinforce' && !p.baseline
          ? 'The baseline is switched off, so plain REINFORCE uses no value function.'
          : 'Darker olive means the network expects to keep the pole up for longer. Leaning and falling further is worth less.';
      if (!agent.steps) status.textContent = method === 'dqn' ? 'The network starts random and ε = 1. Training takes roughly 20–50k steps.' : 'The policy starts uniform: each push is a coin flip. Press Run.';
      else if (avg >= 475) status.innerHTML = '<b>Solved</b>: recent episodes balance the pole for (nearly) all 500 steps.';
      else if (method === 'ac' && p.n === 1 && agent.episodes > 300 && avg < 15) status.innerHTML = 'With n = 1 the critic hardly ever sees a fall 20 ms ahead, its values blow up and the actor collapses to one push. Try n ≥ 8.';
      else status.textContent = run.running ? 'Training…' : 'Paused.';

      const sm = UI.smooth(lens, 20);
      const t = UI.thin(Array.from(sm), 600);
      const dots = UI.thin(lens, 1200);
      p1.set({
        x: { min: 1, max: Math.max(20, n), integer: true },
        y: { min: 0, max: 520 },
        series: [
          { kind: 'line', xs: [1, Math.max(20, n)], ys: [500, 500], color: ink, dash: [4, 4], width: 1, alpha: 0.6 },
          { kind: 'markers', xs: dots.xs, ys: dots.ys, color: muted, radius: n > 400 ? 1.2 : 2, alpha: 0.6 },
          { kind: 'line', xs: t.xs, ys: t.ys, color: accent, width: 2 },
        ],
        message: n ? null : 'No episodes yet',
      });
    }

    // Greedy demo in real time (50 Hz, the simulator's own clock).
    const demoStepper = UI.makeStepper();
    function demoStep() {
      if (!demo) {
        const env = new RL.CartPole(RL.mulberry32(RL.randomSeed()));
        env.reset();
        demo = { env, action: null, best: null, pause: 0 };
      }
      if (demo.env.done) {
        demo.pause++;
        if (demo.pause > 40) {
          const best = demo.env.t;
          demo.env.reset();
          demo.pause = 0;
          demo.best = best;
        }
        return;
      }
      demo.action = agent.greedy(demo.env.state);
      demo.env.step(demo.action);
    }

    const trainStepper = UI.makeStepper();
    reset();

    return {
      frame(dt) {
        let changed = false;
        if (run.running) {
          const n = trainStepper.advance(dt, speed.rate(), () => {
            lastAction = agent.step().action;
          }, 14);
          if (n) changed = true;
        }
        if (state.watch === 'greedy') {
          demoStepper.advance(dt, 50, demoStep, 4);
          changed = true;
        }
        if (changed) render();
      },
      redraw: () => render(true),
      retex: L.retex,
      destroy() {},
    };
  }

  window.Views.deep = view;
})();
