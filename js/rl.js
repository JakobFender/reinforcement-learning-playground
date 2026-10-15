// Environments and algorithms, following Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd ed.).
// No DOM access here, so everything runs in Node for the checks in tests/check.js.
(function (root) {
  'use strict';

  // ---------------------------------------------------------------------------
  // Random numbers
  // ---------------------------------------------------------------------------

  function mulberry32(seed) {
    let a = seed >>> 0;
    const rng = function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    let spare = null;
    rng.normal = function () {
      if (spare != null) {
        const v = spare;
        spare = null;
        return v;
      }
      let u = 0;
      while (u === 0) u = rng();
      const r = Math.sqrt(-2 * Math.log(u));
      const t = 2 * Math.PI * rng();
      spare = r * Math.sin(t);
      return r * Math.cos(t);
    };
    rng.int = (n) => Math.floor(rng() * n);
    return rng;
  }

  const randomSeed = () => (Math.random() * 2 ** 32) >>> 0;

  // Index of the largest value, ties broken at random (as the book recommends).
  function argmaxRandom(values, rng) {
    let best = -Infinity;
    let count = 0;
    let pick = 0;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v > best + 1e-12) {
        best = v;
        pick = i;
        count = 1;
      } else if (Math.abs(v - best) <= 1e-12) {
        count++;
        if (rng() * count < 1) pick = i;
      }
    }
    return pick;
  }

  function argmaxAll(values, tol = 1e-9) {
    let best = -Infinity;
    for (const v of values) if (v > best) best = v;
    const out = [];
    for (let i = 0; i < values.length; i++) if (values[i] >= best - tol) out.push(i);
    return out;
  }

  function softmax(prefs, tau = 1) {
    let m = -Infinity;
    for (const p of prefs) if (p > m) m = p;
    const e = prefs.map((p) => Math.exp((p - m) / tau));
    const z = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / z);
  }

  function sampleIndex(probs, rng) {
    let u = rng();
    for (let i = 0; i < probs.length; i++) {
      u -= probs[i];
      if (u < 0) return i;
    }
    return probs.length - 1;
  }

  // ---------------------------------------------------------------------------
  // Chapter 2: k-armed bandits
  // ---------------------------------------------------------------------------

  // The 10-armed testbed (§2.3): q*(a) ~ N(0, 1), rewards ~ N(q*(a), 1).
  // Non-stationary variant (Exercise 2.5): every q*(a) takes an independent N(0, 0.01²) random walk.
  function makeBandit(k, rng, opts = {}) {
    const q = Array.from({ length: k }, () => (opts.nonstationary ? 0 : rng.normal()) + (opts.offset || 0));
    let best = argmaxAll(q)[0];
    return {
      k,
      q,
      optimal: () => best,
      pull(a) {
        const r = q[a] + rng.normal();
        if (opts.nonstationary) {
          for (let i = 0; i < k; i++) q[i] += 0.01 * rng.normal();
          best = 0;
          for (let i = 1; i < k; i++) if (q[i] > q[best]) best = i;
        }
        return r;
      },
    };
  }

  const banditMethods = {
    egreedy: {
      name: 'ε-greedy',
      defaults: { epsilon: 0.1, alpha: 0, q0: 0 },
    },
    optimistic: {
      name: 'Optimistic initial values',
      defaults: { epsilon: 0, alpha: 0.1, q0: 5 },
    },
    ucb: {
      name: 'Upper confidence bound',
      defaults: { c: 2, alpha: 0, q0: 0 },
    },
    gradient: {
      name: 'Gradient bandit',
      defaults: { alpha: 0.1, baseline: true },
    },
  };

  // alpha = 0 means sample averages (step size 1/n).
  function createBanditAgent(method, p, k, rng) {
    const Q = new Float64Array(k).fill(p.q0 || 0);
    const N = new Float64Array(k);
    const H = new Float64Array(k);
    let t = 0;
    let rBar = 0;
    let pi = new Array(k).fill(1 / k);
    return {
      Q,
      N,
      H,
      get pi() {
        return pi;
      },
      act() {
        t++;
        if (method === 'gradient') {
          pi = softmax(Array.from(H));
          return sampleIndex(pi, rng);
        }
        if (method === 'ucb') {
          for (let a = 0; a < k; a++) if (N[a] === 0) return a;
          const scores = Array.from(Q, (q, a) => q + p.c * Math.sqrt(Math.log(t) / N[a]));
          return argmaxRandom(scores, rng);
        }
        if (rng() < p.epsilon) return rng.int(k);
        return argmaxRandom(Q, rng);
      },
      update(a, r) {
        N[a]++;
        if (method === 'gradient') {
          // §2.8: H(a) ← H(a) + α (R − R̄)(1[a = A] − π(a)); R̄ is the incremental average of all rewards.
          const base = p.baseline ? rBar : 0;
          for (let b = 0; b < k; b++) H[b] += p.alpha * (r - base) * ((b === a ? 1 : 0) - pi[b]);
          rBar += (r - rBar) / t;
          Q[a] += (r - Q[a]) / N[a];
          return;
        }
        const step = p.alpha > 0 ? p.alpha : 1 / N[a];
        Q[a] += step * (r - Q[a]);
      },
    };
  }

  // Average reward and % optimal action over many independent runs (Figures 2.2–2.5).
  function runBandits(method, params, opts = {}) {
    const k = opts.k || 10;
    const steps = opts.steps || 1000;
    const runs = opts.runs || 200;
    const rng = mulberry32(opts.seed == null ? 1 : opts.seed);
    const avgReward = new Float64Array(steps);
    const optimal = new Float64Array(steps);
    let last = null;
    for (let run = 0; run < runs; run++) {
      const bandit = makeBandit(k, rng, { nonstationary: opts.nonstationary, offset: opts.offset });
      const agent = createBanditAgent(method, params, k, rng);
      for (let t = 0; t < steps; t++) {
        const best = bandit.optimal();
        const a = agent.act();
        const r = bandit.pull(a);
        agent.update(a, r);
        avgReward[t] += r;
        if (a === best) optimal[t]++;
      }
      last = { q: bandit.q.slice(), Q: Array.from(agent.Q), N: Array.from(agent.N), pi: agent.pi.slice(), H: Array.from(agent.H) };
    }
    for (let t = 0; t < steps; t++) {
      avgReward[t] /= runs;
      optimal[t] /= runs;
    }
    return { avgReward, optimal, last };
  }

  // ---------------------------------------------------------------------------
  // Gridworlds (Chapters 3–6, 8)
  // ---------------------------------------------------------------------------

  // Cells: '.' free, '#' wall, 'S' start, 'G' goal (terminal), 'X' cliff (−100, back to start).
  const ACTIONS = [
    { name: 'up', dr: -1, dc: 0, arrow: '↑' },
    { name: 'right', dr: 0, dc: 1, arrow: '→' },
    { name: 'down', dr: 1, dc: 0, arrow: '↓' },
    { name: 'left', dr: 0, dc: -1, arrow: '←' },
  ];

  const gridPresets = {
    cliff: {
      name: 'Cliff walking',
      ref: 'Example 6.6',
      rows: ['............', '............', '............', 'SXXXXXXXXXXG'],
      gamma: 1,
    },
    windy: {
      name: 'Windy gridworld',
      ref: 'Example 6.5',
      rows: ['..........', '..........', '..........', 'S......G..', '..........', '..........', '..........'],
      wind: [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
      gamma: 1,
    },
    maze: {
      name: 'Dyna maze',
      ref: 'Example 8.1',
      rows: ['.......#G', '..#....#.', 'S.#....#.', '..#......', '.....#...', '.........'],
      gamma: 0.95,
    },
    small: {
      name: 'Small gridworld',
      ref: 'Example 4.1',
      rows: ['G...', '....', '....', '...G'],
      gamma: 1,
    },
    rooms: {
      name: 'Two rooms',
      ref: 'with walls and a pit',
      rows: ['.....#....G', '.....#.....', '..........X', '.....#.....', 'S....#.....'],
      gamma: 0.95,
    },
  };

  class GridWorld {
    constructor(layout) {
      this.rows = layout.rows.length;
      this.cols = layout.rows[0].length;
      this.cells = layout.rows.map((r) => r.split(''));
      this.wind = layout.wind ? layout.wind.slice() : new Array(this.cols).fill(0);
      this.slip = layout.slip || 0;
      this.stepReward = layout.stepReward == null ? -1 : layout.stepReward;
      this.cliffReward = layout.cliffReward == null ? -100 : layout.cliffReward;
      this.rebuild();
    }

    static fromPreset(key, overrides = {}) {
      return new GridWorld(Object.assign({}, gridPresets[key], overrides));
    }

    clone() {
      const g = new GridWorld({ rows: this.cells.map((r) => r.join('')), wind: this.wind, slip: this.slip, stepReward: this.stepReward, cliffReward: this.cliffReward });
      return g;
    }

    rebuild() {
      this.nS = this.rows * this.cols;
      this.nA = 4;
      this.starts = [];
      for (let s = 0; s < this.nS; s++) if (this.cell(s) === 'S') this.starts.push(s);
    }

    cell(s) {
      return this.cells[Math.floor(s / this.cols)][s % this.cols];
    }

    setCell(r, c, ch) {
      if (ch === 'S') {
        // Only one start cell.
        for (const row of this.cells) for (let j = 0; j < row.length; j++) if (row[j] === 'S') row[j] = '.';
      }
      this.cells[r][c] = ch;
      this.rebuild();
    }

    isWall(s) {
      return this.cell(s) === '#';
    }

    isTerminal(s) {
      return this.cell(s) === 'G';
    }

    // States an agent can occupy (everything but walls; cliff cells are never occupied but harmless).
    isState(s) {
      const ch = this.cell(s);
      return ch !== '#' && ch !== 'X';
    }

    hasGoal() {
      for (let s = 0; s < this.nS; s++) if (this.isTerminal(s)) return true;
      return false;
    }

    // Episodes start at 'S', or at a random non-terminal state if there is none (Example 4.1 has no start).
    startState(rng) {
      if (this.starts.length) return this.starts[0];
      const free = [];
      for (let s = 0; s < this.nS; s++) if (this.isState(s) && !this.isTerminal(s)) free.push(s);
      return free[rng.int(free.length)];
    }

    // Deterministic effect of an action, including wind (applied from the column you leave, Example 6.5).
    move(s, a) {
      const r0 = Math.floor(s / this.cols);
      const c0 = s % this.cols;
      const { dr, dc } = ACTIONS[a];
      let r = r0 + dr;
      let c = c0 + dc;
      if (r < 0 || r >= this.rows || c < 0 || c >= this.cols || this.cells[r][c] === '#') {
        r = r0;
        c = c0;
      }
      const w = this.wind[c0] || 0;
      for (let i = 0; i < w; i++) {
        if (r - 1 < 0 || this.cells[r - 1][c] === '#') break;
        r--;
      }
      const s2 = r * this.cols + c;
      if (this.cells[r][c] === 'X') {
        return { s2: this.startOrSelf(s), r: this.cliffReward, done: false, fell: true };
      }
      return { s2, r: this.stepReward, done: this.cells[r][c] === 'G' };
    }

    startOrSelf(s) {
      return this.starts.length ? this.starts[0] : s;
    }

    // Full model p(s', r | s, a): with probability `slip` the action is replaced by a uniformly random one.
    transitions(s, a) {
      if (!this.slip) return [Object.assign({ p: 1 }, this.move(s, a))];
      const out = [];
      for (let b = 0; b < 4; b++) {
        const p = (b === a ? 1 - this.slip : 0) + this.slip / 4;
        out.push(Object.assign({ p }, this.move(s, b)));
      }
      return out;
    }

    step(s, a, rng) {
      let b = a;
      if (this.slip && rng() < this.slip) b = rng.int(4);
      return this.move(s, b);
    }
  }

  // ---------------------------------------------------------------------------
  // Chapter 4: dynamic programming
  // ---------------------------------------------------------------------------

  function uniformPolicy(env) {
    return Array.from({ length: env.nS }, () => [0.25, 0.25, 0.25, 0.25]);
  }

  function qFromV(env, V, gamma, s) {
    const q = [0, 0, 0, 0];
    for (let a = 0; a < 4; a++) {
      for (const t of env.transitions(s, a)) q[a] += t.p * (t.r + (t.done ? 0 : gamma * V[t.s2]));
    }
    return q;
  }

  // One sweep of iterative policy evaluation (§4.1). Returns Δ = max |v_new − v_old|.
  // In place (the book's default) uses new values as soon as they are available; otherwise two arrays.
  function evaluationSweep(env, policy, V, gamma, inPlace = true) {
    let delta = 0;
    const src = inPlace ? V : Float64Array.from(V);
    for (let s = 0; s < env.nS; s++) {
      if (!env.isState(s) || env.isTerminal(s)) continue;
      const q = qFromV(env, src, gamma, s);
      let v = 0;
      for (let a = 0; a < 4; a++) v += policy[s][a] * q[a];
      delta = Math.max(delta, Math.abs(v - V[s]));
      V[s] = v;
    }
    return delta;
  }

  // Greedy policy improvement (§4.2). Ties share probability equally, so arrows show every greedy action.
  function greedyPolicy(env, V, gamma) {
    return Array.from({ length: env.nS }, (_, s) => {
      if (!env.isState(s) || env.isTerminal(s)) return [0.25, 0.25, 0.25, 0.25];
      const best = argmaxAll(qFromV(env, V, gamma, s), 1e-7);
      const p = [0, 0, 0, 0];
      for (const a of best) p[a] = 1 / best.length;
      return p;
    });
  }

  function samePolicy(p1, p2) {
    for (let s = 0; s < p1.length; s++) for (let a = 0; a < 4; a++) if (Math.abs(p1[s][a] - p2[s][a]) > 1e-9) return false;
    return true;
  }

  // One sweep of value iteration (§4.4).
  function valueIterationSweep(env, V, gamma, inPlace = true) {
    let delta = 0;
    const src = inPlace ? V : Float64Array.from(V);
    for (let s = 0; s < env.nS; s++) {
      if (!env.isState(s) || env.isTerminal(s)) continue;
      const v = Math.max(...qFromV(env, src, gamma, s));
      delta = Math.max(delta, Math.abs(v - V[s]));
      V[s] = v;
    }
    return delta;
  }

  // Converged optimal values (used by the planning tab as a stand-in for a trained value network).
  function solveOptimal(env, gamma, theta = 1e-6, maxSweeps = 5000) {
    const V = new Float64Array(env.nS);
    for (let i = 0; i < maxSweeps; i++) if (valueIterationSweep(env, V, gamma) < theta) break;
    return V;
  }

  // ---------------------------------------------------------------------------
  // Chapter 8.11: Monte Carlo tree search (with an AlphaZero-style variant)
  // ---------------------------------------------------------------------------

  // UCT (Kocsis & Szepesvári 2006) as described in §8.11: selection, expansion, simulation, backup.
  // The 'alphazero' variant replaces rollouts with a value estimate and selects with PUCT using a prior.
  // Q values are normalized with the min/max seen in the tree so `c` does not depend on the reward scale.
  function mctsSearch(env, root, opts, rng) {
    const gamma = opts.gamma;
    const nodes = new Map();
    const visits = new Float64Array(env.nS);
    let qMin = Infinity;
    let qMax = -Infinity;
    const norm = (q) => (qMax > qMin ? (q - qMin) / (qMax - qMin) : 0.5);

    function makeNode(s) {
      const node = { s, N: 0, n: [0, 0, 0, 0], W: [0, 0, 0, 0], kids: [null, null, null, null], prior: null };
      if (opts.mode === 'alphazero') {
        const q = qFromV(env, opts.valueNet, gamma, s);
        node.prior = softmax(q, opts.tau || 1);
      }
      return node;
    }

    const rootNode = makeNode(root);
    nodes.set(root, rootNode);

    function select(node) {
      let best = -Infinity;
      let pick = 0;
      for (let a = 0; a < 4; a++) {
        const n = node.n[a];
        let score;
        if (opts.mode === 'alphazero') {
          const q = n ? norm(node.W[a] / n) : 0;
          score = q + opts.c * node.prior[a] * Math.sqrt(node.N + 1) / (1 + n);
        } else {
          if (n === 0) score = 1e9 + rng();
          else score = norm(node.W[a] / n) + opts.c * Math.sqrt(Math.log(node.N) / n);
        }
        if (score > best) {
          best = score;
          pick = a;
        }
      }
      return pick;
    }

    function leafValue(s) {
      if (opts.mode === 'alphazero') return opts.valueNet[s];
      let g = 0;
      let disc = 1;
      let cur = s;
      for (let d = 0; d < opts.rolloutDepth; d++) {
        const t = env.step(cur, rng.int(4), rng);
        g += disc * t.r;
        disc *= gamma;
        visits[t.s2] += 0.25;
        if (t.done) return g;
        cur = t.s2;
      }
      return g;
    }

    for (let sim = 0; sim < opts.simulations; sim++) {
      // Selection + expansion: descend while the chosen edge already has a child node.
      const path = [];
      let node = rootNode;
      let value = 0;
      for (let depth = 0; depth < opts.maxDepth; depth++) {
        const a = select(node);
        const t = env.step(node.s, a, rng);
        path.push({ node, a, r: t.r });
        visits[t.s2] += 1;
        if (t.done) {
          value = 0;
          break;
        }
        if (!node.kids[a] || node.kids[a].s !== t.s2) {
          // New (or, for stochastic transitions, different) outcome: expand and evaluate.
          const child = makeNode(t.s2);
          if (!node.kids[a]) node.kids[a] = child;
          value = leafValue(t.s2);
          break;
        }
        node = node.kids[a];
        if (depth === opts.maxDepth - 1) value = leafValue(node.s);
      }
      // Backup.
      for (let i = path.length - 1; i >= 0; i--) {
        const { node: nd, a, r } = path[i];
        value = r + gamma * value;
        nd.N++;
        nd.n[a]++;
        nd.W[a] += value;
        const q = nd.W[a] / nd.n[a];
        if (q < qMin) qMin = q;
        if (q > qMax) qMax = q;
      }
    }

    const Q = rootNode.W.map((w, a) => (rootNode.n[a] ? w / rootNode.n[a] : NaN));
    const action = argmaxRandom(rootNode.n, rng);
    return { action, counts: rootNode.n.slice(), Q, prior: rootNode.prior, visits };
  }

  // A noisy copy of v* that plays the role of AlphaZero's learned value network.
  function noisyValueNet(env, gamma, sigma, rng) {
    const V = solveOptimal(env, gamma);
    let scale = 0;
    for (let s = 0; s < env.nS; s++) if (env.isState(s)) scale = Math.max(scale, Math.abs(V[s]));
    const out = new Float64Array(env.nS);
    for (let s = 0; s < env.nS; s++) out[s] = env.isTerminal(s) ? 0 : V[s] + sigma * Math.max(1, scale) * rng.normal();
    return out;
  }

  // ---------------------------------------------------------------------------
  // Chapters 5–6: tabular model-free control
  // ---------------------------------------------------------------------------

  // Step-by-step agent so the UI can animate single steps or run thousands per frame.
  //   'mc'        on-policy first-visit Monte Carlo control for ε-soft policies (§5.4)
  //   'sarsa'     on-policy TD control (§6.4)
  //   'expected'  Expected Sarsa (§6.6)
  //   'qlearning' off-policy TD control (§6.5)
  function createTabularAgent(env, method, params, rng) {
    const nS = env.nS;
    const Q = Array.from({ length: nS }, () => new Float64Array(4));
    const counts = Array.from({ length: nS }, () => new Float64Array(4));
    const visits = new Float64Array(nS);
    const agent = {
      Q,
      visits,
      params,
      episodes: 0,
      totalSteps: 0,
      returns: [],
      lengths: [],
      s: null,
      a: null,
      epReturn: 0,
      epSteps: 0,
      trajectory: [],
      lastEpisode: [],
    };

    function policy(s) {
      const eps = params.epsilon;
      const best = argmaxAll(Q[s], 1e-12);
      const p = [eps / 4, eps / 4, eps / 4, eps / 4];
      for (const a of best) p[a] += (1 - eps) / best.length;
      return p;
    }
    agent.policy = policy;

    function choose(s) {
      if (rng() < params.epsilon) return rng.int(4);
      return argmaxRandom(Q[s], rng);
    }

    function startEpisode() {
      agent.s = env.startState(rng);
      agent.a = choose(agent.s);
      agent.epReturn = 0;
      agent.epSteps = 0;
      agent.trajectory = [agent.s];
      agent.history = [];
    }

    function endEpisode(truncated) {
      if (method === 'mc') {
        // First-visit returns, processed backwards (§5.4 box). Constant α if set, else sample averages.
        const first = new Map();
        agent.history.forEach((h, t) => {
          const key = h.s * 4 + h.a;
          if (!first.has(key)) first.set(key, t);
        });
        let G = 0;
        for (let t = agent.history.length - 1; t >= 0; t--) {
          const h = agent.history[t];
          G = params.gamma * G + h.r;
          if (first.get(h.s * 4 + h.a) === t) {
            counts[h.s][h.a]++;
            const step = params.alpha > 0 ? params.alpha : 1 / counts[h.s][h.a];
            Q[h.s][h.a] += step * (G - Q[h.s][h.a]);
          }
        }
      }
      agent.episodes++;
      agent.returns.push(agent.epReturn);
      agent.lengths.push(agent.epSteps);
      agent.lastEpisode = agent.trajectory;
      agent.lastTruncated = truncated;
      agent.s = null;
    }

    // Returns info about the transition just taken.
    agent.step = function () {
      if (agent.s == null) startEpisode();
      const s = agent.s;
      const a = agent.a;
      const t = env.step(s, a, rng);
      visits[s]++;
      agent.totalSteps++;
      agent.epSteps++;
      agent.epReturn += t.r;
      agent.trajectory.push(t.s2);
      const s2 = t.s2;
      let a2 = null;
      if (method === 'mc') {
        agent.history.push({ s, a, r: t.r });
        if (!t.done) a2 = choose(s2);
      } else {
        const g = params.gamma;
        let target = t.r;
        if (!t.done) {
          if (method === 'sarsa') {
            a2 = choose(s2);
            target += g * Q[s2][a2];
          } else if (method === 'expected') {
            const p = policy(s2);
            let ev = 0;
            for (let b = 0; b < 4; b++) ev += p[b] * Q[s2][b];
            target += g * ev;
            a2 = choose(s2);
          } else {
            target += g * Math.max(...Q[s2]);
            a2 = choose(s2);
          }
        }
        Q[s][a] += params.alpha * (target - Q[s][a]);
      }
      const truncated = !t.done && agent.epSteps >= params.maxSteps;
      const info = { s, a, r: t.r, s2, done: t.done || truncated, fell: t.fell };
      if (t.done || truncated) endEpisode(truncated);
      else {
        agent.s = s2;
        agent.a = a2;
      }
      return info;
    };

    agent.runEpisode = function (cap = 1e6) {
      let n = 0;
      do {
        n++;
      } while (!agent.step().done && n < cap);
    };

    agent.greedyPath = function (limit = 200) {
      const path = [];
      let s = env.startState(mulberry32(1));
      path.push(s);
      const seen = new Set([s]);
      for (let i = 0; i < limit; i++) {
        const a = argmaxAll(Q[s])[0];
        const t = env.move(s, a);
        path.push(t.s2);
        if (t.done || seen.has(t.s2)) break;
        seen.add(t.s2);
        s = t.s2;
      }
      return path;
    };

    return agent;
  }

  // ---------------------------------------------------------------------------
  // Chapter 6: TD prediction on the random walk (Example 6.2)
  // ---------------------------------------------------------------------------

  // States 0 and 6 are terminal; A–E are 1..5; start in C; +1 for exiting right. True values 1/6 … 5/6.
  const RANDOM_WALK_TRUE = [0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 0];

  function randomWalkEpisode(rng) {
    const states = [3];
    let s = 3;
    while (s !== 0 && s !== 6) {
      s += rng() < 0.5 ? -1 : 1;
      states.push(s);
    }
    return { states, reward: s === 6 ? 1 : 0 };
  }

  // Update V in place from one episode. method: 'td' (TD(0), §6.1) or 'mc' (constant-α MC, eq. 6.1). γ = 1.
  function randomWalkUpdate(V, ep, method, alpha) {
    const { states, reward } = ep;
    if (method === 'td') {
      for (let i = 0; i < states.length - 1; i++) {
        const s = states[i];
        const s2 = states[i + 1];
        const r = s2 === 6 ? 1 : 0;
        const v2 = s2 === 0 || s2 === 6 ? 0 : V[s2];
        V[s] += alpha * (r + v2 - V[s]);
      }
    } else {
      for (let i = 0; i < states.length - 1; i++) {
        const s = states[i];
        V[s] += alpha * (reward - V[s]);
      }
    }
  }

  function rmsError(V) {
    let e = 0;
    for (let s = 1; s <= 5; s++) e += (V[s] - RANDOM_WALK_TRUE[s]) ** 2;
    return Math.sqrt(e / 5);
  }

  // RMS error (averaged over states, then over runs) after each episode — Example 6.2, right panel.
  function randomWalkCurve(method, alpha, episodes, runs, seed = 1) {
    const rng = mulberry32(seed);
    const out = new Float64Array(episodes + 1);
    for (let run = 0; run < runs; run++) {
      const V = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0];
      out[0] += rmsError(V);
      for (let e = 1; e <= episodes; e++) {
        randomWalkUpdate(V, randomWalkEpisode(rng), method, alpha);
        out[e] += rmsError(V);
      }
    }
    for (let e = 0; e <= episodes; e++) out[e] /= runs;
    return out;
  }

  // ---------------------------------------------------------------------------
  // Function approximation: cart-pole (Barto, Sutton & Anderson 1983; Example 3.4)
  // ---------------------------------------------------------------------------

  const CARTPOLE = {
    gravity: 9.8,
    massCart: 1.0,
    massPole: 0.1,
    halfLength: 0.5,
    force: 10,
    tau: 0.02,
    xLimit: 2.4,
    thetaLimit: (12 * Math.PI) / 180,
    maxSteps: 500,
  };

  class CartPole {
    constructor(rng) {
      this.rng = rng;
      this.reset();
    }

    reset() {
      const u = () => (this.rng() - 0.5) * 0.1;
      this.state = [u(), u(), u(), u()];
      this.t = 0;
      this.done = false;
      return this.state.slice();
    }

    // Action 0 pushes left, 1 pushes right. Reward +1 per step until the pole falls (or 500 steps).
    step(action) {
      const P = CARTPOLE;
      let [x, xDot, theta, thetaDot] = this.state;
      const force = action === 1 ? P.force : -P.force;
      const cos = Math.cos(theta);
      const sin = Math.sin(theta);
      const total = P.massCart + P.massPole;
      const pml = P.massPole * P.halfLength;
      const temp = (force + pml * thetaDot * thetaDot * sin) / total;
      const thetaAcc = (P.gravity * sin - cos * temp) / (P.halfLength * (4 / 3 - (P.massPole * cos * cos) / total));
      const xAcc = temp - (pml * thetaAcc * cos) / total;
      x += P.tau * xDot;
      xDot += P.tau * xAcc;
      theta += P.tau * thetaDot;
      thetaDot += P.tau * thetaAcc;
      this.state = [x, xDot, theta, thetaDot];
      this.t++;
      const failed = Math.abs(x) > P.xLimit || Math.abs(theta) > P.thetaLimit;
      const truncated = !failed && this.t >= P.maxSteps;
      this.done = failed || truncated;
      return { state: this.state.slice(), reward: 1, done: this.done, failed, truncated };
    }
  }

  // Inputs scaled to roughly [−1, 1].
  function cartFeatures(st) {
    return [st[0] / 2.4, st[1] / 3, st[2] / 0.21, st[3] / 3];
  }

  // ---------------------------------------------------------------------------
  // A tiny multilayer perceptron with Adam
  // ---------------------------------------------------------------------------

  class MLP {
    constructor(sizes, rng, opts = {}) {
      this.sizes = sizes;
      this.layers = [];
      for (let i = 0; i < sizes.length - 1; i++) {
        const nin = sizes[i];
        const nout = sizes[i + 1];
        const last = i === sizes.length - 2;
        const scale = last ? (opts.outScale == null ? 0.1 : opts.outScale) * Math.sqrt(1 / nin) : Math.sqrt(2 / nin);
        const W = new Float64Array(nin * nout);
        for (let j = 0; j < W.length; j++) W[j] = rng.normal() * scale;
        this.layers.push({
          nin,
          nout,
          W,
          b: new Float64Array(nout),
          gW: new Float64Array(nin * nout),
          gb: new Float64Array(nout),
          mW: new Float64Array(nin * nout),
          vW: new Float64Array(nin * nout),
          mb: new Float64Array(nout),
          vb: new Float64Array(nout),
        });
      }
      this.t = 0;
      this.acts = [];
    }

    // ReLU hidden layers, linear output. Keeps activations for a following backward().
    // Returns an internal buffer that the next forward() overwrites.
    forward(x) {
      if (!this.bufs) {
        this.bufs = this.sizes.map((n) => new Float64Array(n));
        this.dbufs = this.sizes.map((n) => new Float64Array(n));
      }
      const acts = this.bufs;
      for (let i = 0; i < x.length; i++) acts[0][i] = x[i];
      const nL = this.layers.length;
      for (let li = 0; li < nL; li++) {
        const L = this.layers[li];
        const h = acts[li];
        const out = acts[li + 1];
        const last = li === nL - 1;
        const W = L.W;
        const nin = L.nin;
        for (let o = 0; o < L.nout; o++) {
          let z = L.b[o];
          const off = o * nin;
          for (let i = 0; i < nin; i++) z += W[off + i] * h[i];
          out[o] = last || z > 0 ? z : 0;
        }
      }
      return acts[nL];
    }

    // Accumulates dLoss/dparams given dLoss/doutput for the last forward() input.
    backward(dout) {
      const acts = this.bufs;
      const ds = this.dbufs;
      for (let i = 0; i < dout.length; i++) ds[this.layers.length][i] = dout[i];
      for (let li = this.layers.length - 1; li >= 0; li--) {
        const L = this.layers[li];
        const hin = acts[li];
        const d = ds[li + 1];
        const din = ds[li];
        const nin = L.nin;
        const W = L.W;
        const gW = L.gW;
        din.fill(0);
        for (let o = 0; o < L.nout; o++) {
          const g = d[o];
          if (g === 0) continue;
          L.gb[o] += g;
          const off = o * nin;
          for (let i = 0; i < nin; i++) {
            gW[off + i] += g * hin[i];
            din[i] += g * W[off + i];
          }
        }
        if (li > 0) for (let i = 0; i < nin; i++) if (hin[i] <= 0) din[i] = 0;
      }
    }

    _gradScale(scale, clip) {
      if (!(clip > 0)) return scale;
      let norm = 0;
      for (const L of this.layers) {
        for (const g of L.gW) norm += g * g;
        for (const g of L.gb) norm += g * g;
      }
      norm = Math.sqrt(norm) * Math.abs(scale);
      return norm > clip ? (scale * clip) / norm : scale;
    }

    // Plain stochastic gradient descent, as in the book's pseudocode.
    sgd(lr, scale = 1, clip = 0) {
      const k = this._gradScale(scale, clip) * lr;
      for (const L of this.layers) {
        for (let i = 0; i < L.W.length; i++) {
          L.W[i] -= k * L.gW[i];
          L.gW[i] = 0;
        }
        for (let i = 0; i < L.b.length; i++) {
          L.b[i] -= k * L.gb[i];
          L.gb[i] = 0;
        }
      }
    }

    // Adam (Kingma & Ba 2015) on the accumulated gradients, then clears them.
    step(lr, scale = 1, clip = 0) {
      this.t++;
      const b1 = 0.9;
      const b2 = 0.999;
      const c1 = 1 - Math.pow(b1, this.t);
      const c2 = 1 - Math.pow(b2, this.t);
      const k = this._gradScale(scale, clip);
      const upd = (P, G, M, V) => {
        for (let i = 0; i < P.length; i++) {
          const g = G[i] * k;
          M[i] = b1 * M[i] + (1 - b1) * g;
          V[i] = b2 * V[i] + (1 - b2) * g * g;
          P[i] -= (lr * (M[i] / c1)) / (Math.sqrt(V[i] / c2) + 1e-8);
          G[i] = 0;
        }
      };
      for (const L of this.layers) {
        upd(L.W, L.gW, L.mW, L.vW);
        upd(L.b, L.gb, L.mb, L.vb);
      }
    }

    copyFrom(other) {
      this.layers.forEach((L, i) => {
        L.W.set(other.layers[i].W);
        L.b.set(other.layers[i].b);
      });
    }

    // Polyak averaging: θ⁻ ← (1 − τ) θ⁻ + τ θ.
    blend(other, tau) {
      this.layers.forEach((L, i) => {
        const O = other.layers[i];
        for (let j = 0; j < L.W.length; j++) L.W[j] += tau * (O.W[j] - L.W[j]);
        for (let j = 0; j < L.b.length; j++) L.b[j] += tau * (O.b[j] - L.b[j]);
      });
    }

    clone(rng) {
      const m = new MLP(this.sizes, rng);
      m.copyFrom(this);
      return m;
    }
  }

  // ---------------------------------------------------------------------------
  // Deep Q-learning (Mnih et al. 2015; semi-gradient Q-learning, §10.1 + §16.5)
  // ---------------------------------------------------------------------------

  const dqnDefaults = {
    hidden: 32,
    lr: 0.001,
    gamma: 0.99,
    batch: 32,
    buffer: 20000,
    targetEvery: 500,
    epsStart: 1,
    epsEnd: 0.02,
    epsDecay: 8000,
    warmup: 500,
    useTarget: true,
    useReplay: true,
    double: false,
    failReward: true,
    tau: 0.005,
    trainEvery: 1,
  };

  function createDQN(params, rng) {
    const p = Object.assign({}, dqnDefaults, params);
    const net = new MLP([4, p.hidden, p.hidden, 2], rng);
    const target = net.clone(rng);
    const buf = [];
    let bufPos = 0;
    const env = new CartPole(rng);
    const agent = {
      p,
      net,
      env,
      steps: 0,
      episodes: 0,
      lengths: [],
      losses: [],
      lossAcc: 0,
      lossN: 0,
      state: env.reset(),
      epLen: 0,
    };
    agent.epsilon = () => (p.epsDecay > 0 ? Math.max(p.epsEnd, p.epsStart - ((p.epsStart - p.epsEnd) * agent.steps) / p.epsDecay) : p.epsEnd);
    agent.q = (st) => net.forward(cartFeatures(st));
    agent.greedy = (st) => {
      const q = agent.q(st);
      return q[1] > q[0] ? 1 : 0;
    };

    function remember(tr) {
      if (buf.length < p.buffer) buf.push(tr);
      else buf[bufPos] = tr;
      bufPos = (bufPos + 1) % p.buffer;
    }

    function learn(batch) {
      const tnet = p.useTarget ? target : net;
      let loss = 0;
      // Targets first (with the target network), then gradients of the Huber loss.
      const ys = batch.map((tr) => {
        if (tr.terminal) return tr.r;
        if (p.double && p.useTarget) {
          // Double DQN (van Hasselt et al. 2016; §6.7): the online net picks, the target net evaluates.
          const qo = net.forward(tr.f2);
          const a2 = qo[1] > qo[0] ? 1 : 0;
          return tr.r + p.gamma * tnet.forward(tr.f2)[a2];
        }
        const q2 = tnet.forward(tr.f2);
        return tr.r + p.gamma * Math.max(q2[0], q2[1]);
      });
      batch.forEach((tr, i) => {
        const q = net.forward(tr.f);
        const err = q[tr.a] - ys[i];
        const g = Math.max(-1, Math.min(1, err));
        loss += Math.abs(err) <= 1 ? 0.5 * err * err : Math.abs(err) - 0.5;
        const d = [0, 0];
        d[tr.a] = g;
        net.backward(d);
      });
      net.step(p.lr, 1 / batch.length);
      return loss / batch.length;
    }

    agent.step = function () {
      const st = agent.state;
      const f = cartFeatures(st);
      const a = rng() < agent.epsilon() ? rng.int(2) : agent.greedy(st);
      const t = env.step(a);
      // Hitting the 500-step limit is a time-out, not a terminal state: keep bootstrapping.
      // Example 3.4 offers two rewards: +1 per step, or −1 on failure and 0 otherwise (discounted).
      // The second keeps Q in [−1, 0], which suits a squared/Huber loss much better than Q ≈ 100.
      const r = p.failReward ? (t.failed ? -1 : 0) : t.reward;
      const tr = { f, a, r, f2: cartFeatures(t.state), terminal: t.failed };
      agent.steps++;
      agent.epLen++;
      if (p.useReplay) {
        remember(tr);
        if (buf.length >= Math.max(p.warmup, p.batch) && agent.steps % p.trainEvery === 0) {
          const batch = [];
          for (let i = 0; i < p.batch; i++) batch.push(buf[rng.int(buf.length)]);
          agent.lossAcc += learn(batch);
          agent.lossN++;
        }
      } else {
        agent.lossAcc += learn([tr]);
        agent.lossN++;
      }
      if (p.useTarget) {
        if (p.tau > 0) target.blend(net, p.tau);
        else if (agent.steps % p.targetEvery === 0) target.copyFrom(net);
      }
      const info = { state: t.state, action: a, done: t.done };
      if (t.done) {
        agent.episodes++;
        agent.lengths.push(agent.epLen);
        agent.losses.push(agent.lossN ? agent.lossAcc / agent.lossN : NaN);
        agent.lossAcc = 0;
        agent.lossN = 0;
        agent.epLen = 0;
        agent.state = env.reset();
      } else agent.state = t.state;
      return info;
    };

    agent.policyProb = (st) => (agent.greedy(st) === 1 ? 1 : 0);
    agent.value = (st) => {
      const q = agent.q(st);
      return Math.max(q[0], q[1]);
    };
    return agent;
  }

  // ---------------------------------------------------------------------------
  // Policy gradient: REINFORCE (§13.3) and REINFORCE with baseline (§13.4)
  // ---------------------------------------------------------------------------

  const reinforceDefaults = { hidden: 32, lr: 0.005, lrValue: 0.01, gamma: 0.99, baseline: true, discountFactor: true };

  function policyProbs(net, st) {
    const z = net.forward(cartFeatures(st));
    return softmax([z[0], z[1]]);
  }

  function createReinforce(params, rng) {
    const p = Object.assign({}, reinforceDefaults, params);
    const pi = new MLP([4, p.hidden, 2], rng, { outScale: 0.01 });
    const vnet = new MLP([4, p.hidden, 1], rng);
    const env = new CartPole(rng);
    const agent = { p, pi, vnet, env, steps: 0, episodes: 0, lengths: [], state: env.reset(), epLen: 0, ep: [] };

    agent.prob = (st) => policyProbs(pi, st)[1];
    agent.value = (st) => vnet.forward(cartFeatures(st))[0];

    function finishEpisode() {
      const T = agent.ep.length;
      let G = 0;
      const returns = new Array(T);
      for (let t = T - 1; t >= 0; t--) {
        G = agent.ep[t].r + p.gamma * G;
        returns[t] = G;
      }
      let disc = 1;
      for (let t = 0; t < T; t++) {
        const { f, a } = agent.ep[t];
        let delta = returns[t];
        if (p.baseline) {
          const v = vnet.forward(f)[0];
          delta = returns[t] - v;
          // Value net: gradient of ½(G − v̂)² is −(G − v̂).
          vnet.backward([-delta]);
        }
        const probs = softmax(Array.from(pi.forward(f)));
        const scale = (p.discountFactor ? disc : 1) * delta;
        // Gradient ascent on ln π(a|s): d/dz = 1[a] − π, so the loss gradient is −scale (1[a] − π).
        pi.backward([-scale * ((a === 0 ? 1 : 0) - probs[0]), -scale * ((a === 1 ? 1 : 0) - probs[1])]);
        disc *= p.gamma;
      }
      pi.step(p.lr, 1 / T, 50);
      if (p.baseline) vnet.step(p.lrValue, 1 / T);
      agent.ep = [];
    }

    agent.step = function () {
      const st = agent.state;
      const f = cartFeatures(st);
      const a = rng() < policyProbs(pi, st)[1] ? 1 : 0;
      const t = env.step(a);
      agent.ep.push({ f, a, r: t.reward });
      agent.steps++;
      agent.epLen++;
      const info = { state: t.state, action: a, done: t.done };
      if (t.done) {
        finishEpisode();
        agent.episodes++;
        agent.lengths.push(agent.epLen);
        agent.epLen = 0;
        agent.state = env.reset();
      } else agent.state = t.state;
      return info;
    };
    agent.greedy = (st) => (agent.prob(st) > 0.5 ? 1 : 0);
    return agent;
  }

  // ---------------------------------------------------------------------------
  // Actor–critic (§13.5) with n-step returns (Chapter 7)
  // ---------------------------------------------------------------------------

  // n = 1 is the book's one-step actor–critic. Larger n bootstraps from v̂ further ahead, which helps a lot
  // on cart-pole: consecutive states are only 20 ms apart, so a one-step critic rarely "sees" a failure.
  const acDefaults = { hidden: 32, lrActor: 0.001, lrCritic: 0.003, gamma: 0.99, n: 32, discountFactor: false };

  function createActorCritic(params, rng) {
    const p = Object.assign({}, acDefaults, params);
    const pi = new MLP([4, p.hidden, 2], rng, { outScale: 0.01 });
    const vnet = new MLP([4, p.hidden, 1], rng);
    const env = new CartPole(rng);
    const agent = { p, pi, vnet, env, steps: 0, episodes: 0, lengths: [], state: env.reset(), epLen: 0, I: 1, tdErrors: [], tdAcc: 0, tdN: 0, pending: [] };

    agent.prob = (st) => policyProbs(pi, st)[1];
    agent.value = (st) => vnet.forward(cartFeatures(st))[0];

    // Update the oldest pending step with G = R₁ + γR₂ + … + γⁿ⁻¹Rₙ + γⁿ v̂(Sₙ).
    function updateOldest(bootstrap) {
      const q = agent.pending;
      let G = bootstrap;
      for (let k = q.length - 1; k >= 0; k--) G = q[k].r + p.gamma * G;
      const { f, a } = q.shift();
      const v = vnet.forward(f)[0];
      const delta = G - v;
      vnet.backward([-delta]);
      vnet.step(p.lrCritic, 1, 10);
      const probs = softmax(Array.from(pi.forward(f)));
      const scale = (p.discountFactor ? agent.I : 1) * delta;
      pi.backward([-scale * ((a === 0 ? 1 : 0) - probs[0]), -scale * ((a === 1 ? 1 : 0) - probs[1])]);
      pi.step(p.lrActor, 1, 10);
      agent.I *= p.gamma;
      agent.tdAcc += Math.abs(delta);
      agent.tdN++;
      return delta;
    }

    agent.step = function () {
      const st = agent.state;
      const f = cartFeatures(st);
      const a = rng() < agent.prob(st) ? 1 : 0;
      const t = env.step(a);
      agent.pending.push({ f, a, r: t.reward });
      agent.steps++;
      agent.epLen++;
      let delta = null;
      if (t.done) {
        // A fall ends the return; a time-out still bootstraps from v̂.
        const tail = t.failed ? 0 : agent.value(t.state);
        while (agent.pending.length) delta = updateOldest(tail);
      } else if (agent.pending.length >= p.n) {
        delta = updateOldest(agent.value(t.state));
      }
      const info = { state: t.state, action: a, done: t.done, delta };
      if (t.done) {
        agent.episodes++;
        agent.lengths.push(agent.epLen);
        agent.tdErrors.push(agent.tdN ? agent.tdAcc / agent.tdN : NaN);
        agent.tdAcc = 0;
        agent.tdN = 0;
        agent.epLen = 0;
        agent.I = 1;
        agent.state = env.reset();
      } else agent.state = t.state;
      return info;
    };
    agent.greedy = (st) => (agent.prob(st) > 0.5 ? 1 : 0);
    return agent;
  }

  // Run a greedy policy for one episode (used to report "greedy length" in the UI and in checks).
  function evalCartPole(policy, rng, episodes = 1) {
    let total = 0;
    for (let e = 0; e < episodes; e++) {
      const env = new CartPole(rng);
      let st = env.reset();
      for (;;) {
        const t = env.step(policy(st));
        st = t.state;
        if (t.done) break;
      }
      total += env.t;
    }
    return total / episodes;
  }

  root.RL = {
    mulberry32,
    randomSeed,
    argmaxRandom,
    argmaxAll,
    softmax,
    sampleIndex,
    makeBandit,
    banditMethods,
    createBanditAgent,
    runBandits,
    ACTIONS,
    gridPresets,
    GridWorld,
    uniformPolicy,
    qFromV,
    evaluationSweep,
    greedyPolicy,
    samePolicy,
    valueIterationSweep,
    solveOptimal,
    mctsSearch,
    noisyValueNet,
    createTabularAgent,
    RANDOM_WALK_TRUE,
    randomWalkEpisode,
    randomWalkUpdate,
    rmsError,
    randomWalkCurve,
    CARTPOLE,
    CartPole,
    cartFeatures,
    MLP,
    dqnDefaults,
    createDQN,
    reinforceDefaults,
    createReinforce,
    acDefaults,
    createActorCritic,
    evalCartPole,
  };
})(typeof globalThis !== 'undefined' ? globalThis : this);
