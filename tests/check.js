// Sanity checks for the RL algorithms. Run with: node tests/check.js
'use strict';
require('../js/rl.js');
const { RL } = globalThis;

let failures = 0;
let passed = 0;
function check(cond, msg) {
  if (!cond) {
    failures++;
    console.error('FAIL:', msg);
  } else passed++;
}
const close = (a, b, tol) => Math.abs(a - b) <= tol;
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const tail = (arr, n) => mean(Array.from(arr).slice(-n));

// 1. Bandits reproduce the qualitative results of Chapter 2.
{
  const opts = { runs: 300, steps: 1000, seed: 3 };
  const greedy = RL.runBandits('egreedy', { epsilon: 0, alpha: 0, q0: 0 }, opts);
  const eps = RL.runBandits('egreedy', { epsilon: 0.1, alpha: 0, q0: 0 }, opts);
  check(tail(eps.avgReward, 200) > tail(greedy.avgReward, 200), 'ε = 0.1 earns more than greedy (Figure 2.2)');
  check(tail(eps.optimal, 200) > 0.7 && tail(greedy.optimal, 200) < 0.5, 'ε-greedy finds the best arm far more often');
  const ucb = RL.runBandits('ucb', { c: 2, alpha: 0, q0: 0 }, opts);
  check(tail(ucb.avgReward, 500) > tail(eps.avgReward, 500), 'UCB beats ε-greedy late on (Figure 2.4)');
  const opt = RL.runBandits('optimistic', { epsilon: 0, alpha: 0.1, q0: 5 }, opts);
  const real = RL.runBandits('egreedy', { epsilon: 0.1, alpha: 0.1, q0: 0 }, opts);
  check(tail(opt.optimal, 200) > tail(real.optimal, 200), 'optimistic Q₁ = 5 beats realistic ε-greedy (Figure 2.3)');
  const o4 = Object.assign({ offset: 4 }, opts);
  const withB = RL.runBandits('gradient', { alpha: 0.1, baseline: true }, o4);
  const noB = RL.runBandits('gradient', { alpha: 0.1, baseline: false }, o4);
  check(tail(withB.optimal, 200) > tail(noB.optimal, 200) + 0.2, 'gradient bandit needs the baseline when q* is shifted by +4 (Figure 2.5)');
  const k = 10;
  const agent = RL.createBanditAgent('gradient', { alpha: 0.1, baseline: true }, k, RL.mulberry32(1));
  agent.act();
  check(close(agent.pi.reduce((a, b) => a + b, 0), 1, 1e-12), 'soft-max preferences sum to one');
}

// 2. Example 4.1: iterative policy evaluation of the equiprobable random policy.
{
  const env = RL.GridWorld.fromPreset('small');
  const V = new Float64Array(env.nS);
  const pol = RL.uniformPolicy(env);
  let sweeps = 0;
  while (RL.evaluationSweep(env, pol, V, 1) > 1e-8 && sweeps < 10000) sweeps++;
  const expected = [0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0];
  check(expected.every((v, s) => close(V[s], v, 1e-4)), `v_π matches Figure 4.1 (got ${Array.from(V, (v) => v.toFixed(1)).join(' ')})`);

  // Policy iteration and value iteration agree on v*.
  let policy = RL.uniformPolicy(env);
  const Vpi = new Float64Array(env.nS);
  for (let it = 0; it < 50; it++) {
    while (RL.evaluationSweep(env, policy, Vpi, 1) > 1e-10);
    const next = RL.greedyPolicy(env, Vpi, 1);
    if (RL.samePolicy(next, policy)) break;
    policy = next;
  }
  const Vvi = RL.solveOptimal(env, 1, 1e-12);
  check(Array.from(Vvi).every((v, s) => close(v, Vpi[s], 1e-6)), 'policy iteration and value iteration reach the same v*');
  check(close(Vvi[3], -3, 1e-9) && close(Vvi[1], -1, 1e-9), 'v* is minus the distance to the nearest corner');
}

// 3. Stochastic transitions: probabilities sum to one.
{
  const env = RL.GridWorld.fromPreset('windy', { slip: 0.2 });
  let ok = true;
  for (let s = 0; s < env.nS; s++)
    for (let a = 0; a < 4; a++) if (!close(env.transitions(s, a).reduce((t, x) => t + x.p, 0), 1, 1e-12)) ok = false;
  check(ok, 'transition probabilities sum to one');
  const plain = RL.GridWorld.fromPreset('windy');
  const start = plain.starts[0];
  // Example 6.5: optimal path takes 15 steps.
  const V = RL.solveOptimal(plain, 1);
  check(close(V[start], -15, 1e-9), `windy gridworld shortest path is 15 steps (got ${-V[start]})`);
}

// 4. Example 6.6: Q-learning walks the cliff edge, Sarsa takes the safe path.
{
  const env = RL.GridWorld.fromPreset('cliff');
  const run = (method) => {
    const ag = RL.createTabularAgent(env, method, { alpha: 0.5, epsilon: 0.1, gamma: 1, maxSteps: 1000 }, RL.mulberry32(5));
    for (let e = 0; e < 500; e++) ag.runEpisode();
    return ag;
  };
  const q = run('qlearning');
  const sarsa = run('sarsa');
  const expected = run('expected');
  check(q.greedyPath().length - 1 === 13, `Q-learning greedy path is the 13-step optimal one (got ${q.greedyPath().length - 1})`);
  check(sarsa.greedyPath().length - 1 > 13, 'Sarsa learns a longer, safer path');
  check(tail(sarsa.returns, 100) > tail(q.returns, 100), 'Sarsa earns more online than Q-learning (Figure 6.4)');
  check(tail(expected.returns, 100) > tail(q.returns, 100), 'Expected Sarsa also earns more online than Q-learning');
}

// 5. Monte Carlo control finds the goal of the Dyna maze.
{
  const env = RL.GridWorld.fromPreset('maze');
  const ag = RL.createTabularAgent(env, 'mc', { alpha: 0, epsilon: 0.1, gamma: 0.95, maxSteps: 2000 }, RL.mulberry32(2));
  for (let e = 0; e < 400; e++) ag.runEpisode();
  const path = ag.greedyPath();
  check(env.isTerminal(path[path.length - 1]), 'MC control greedy policy reaches the goal');
  check(path.length - 1 <= 20, `MC control path is reasonably short (got ${path.length - 1}, optimum 14)`);
}

// 6. Example 6.2: TD(0) beats constant-α MC on the random walk.
{
  const td = RL.randomWalkCurve('td', 0.1, 100, 100);
  const mc = RL.randomWalkCurve('mc', 0.02, 100, 100);
  check(td[100] < mc[100], `TD(0) has lower RMS error after 100 episodes (TD ${td[100].toFixed(3)}, MC ${mc[100].toFixed(3)})`);
  check(close(td[0], Math.sqrt((4 / 36 + 1 / 36 + 0 + 1 / 36 + 4 / 36) / 5), 1e-12), 'initial RMS error matches V = 0.5');
}

// 7. MCTS picks a first move along a shortest path in the maze.
{
  const env = RL.GridWorld.fromPreset('maze');
  const Vstar = RL.solveOptimal(env, 0.95);
  const s = env.starts[0];
  const qs = RL.qFromV(env, Vstar, 0.95, s);
  const best = RL.argmaxAll(qs, 1e-9);
  const uct = RL.mctsSearch(env, s, { gamma: 0.95, simulations: 3000, c: 1.4, rolloutDepth: 60, maxDepth: 60, mode: 'uct' }, RL.mulberry32(4));
  check(best.includes(uct.action), `UCT picks an optimal first move (picked ${uct.action}, optimal ${best})`);
  const net = RL.noisyValueNet(env, 0.95, 0.02, RL.mulberry32(1));
  const az = RL.mctsSearch(env, s, { gamma: 0.95, simulations: 200, c: 1.5, maxDepth: 60, mode: 'alphazero', valueNet: net }, RL.mulberry32(4));
  check(best.includes(az.action), 'AlphaZero-style search picks an optimal first move with 200 simulations');
}

// 8. MLP gradients match finite differences.
{
  const rng = RL.mulberry32(11);
  const net = new RL.MLP([4, 8, 3], rng, { outScale: 1 });
  const x = [0.3, -0.2, 0.5, 0.1];
  const w = [0.7, -1.3, 0.4];
  const loss = () => {
    const y = net.forward(x);
    return y[0] * w[0] + y[1] * w[1] + y[2] * w[2];
  };
  loss();
  net.backward(w);
  let worst = 0;
  for (const L of net.layers) {
    for (let i = 0; i < L.W.length; i += 3) {
      const old = L.W[i];
      L.W[i] = old + 1e-6;
      const up = loss();
      L.W[i] = old - 1e-6;
      const dn = loss();
      L.W[i] = old;
      worst = Math.max(worst, Math.abs((up - dn) / 2e-6 - L.gW[i]));
    }
  }
  check(worst < 1e-6, `backprop matches finite differences (max error ${worst.toExponential(1)})`);
}

// 9. Cart-pole learners actually learn (fixed seeds, so this is deterministic).
{
  const physics = new RL.CartPole(RL.mulberry32(1));
  let n = 0;
  while (!physics.step(1).done) n++;
  check(n > 5 && n < 20, 'always pushing right drops the pole quickly');

  const best = (make, seed, steps, every) => {
    const ag = make(RL.mulberry32(seed));
    let top = 0;
    while (ag.steps < steps) {
      ag.step();
      if (ag.steps % every === 0) top = Math.max(top, RL.evalCartPole(ag.greedy, RL.mulberry32(9), 3));
    }
    return top;
  };
  const rf = best((rng) => RL.createReinforce({}, rng), 1, 60000, 10000);
  check(rf >= 450, `REINFORCE with baseline balances the pole (best greedy length ${rf})`);
  const ac = best((rng) => RL.createActorCritic({}, rng), 1, 60000, 10000);
  check(ac >= 450, `n-step actor–critic balances the pole (best greedy length ${ac})`);
  const dqn = best((rng) => RL.createDQN({}, rng), 1, 50000, 5000);
  check(dqn >= 450, `DQN balances the pole (best greedy length ${dqn})`);
}

console.log(`${passed} passed, ${failures} failed`);
process.exit(failures ? 1 : 0);
