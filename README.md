# Reinforcement Learning Playground

An interactive, dependency-free web app with sixteen algorithms from Sutton & Barto's
*Reinforcement Learning: An Introduction* (2nd edition), each running live in the browser.

**Live demo:** https://jakobfender.github.io/reinforcement-learning-playground/ (after enabling Pages, see below)

## Algorithms

Every algorithm is its own tab (linkable, e.g. `#q-learning`), grouped like the usual map of the field:

```
RL
├── Bandits · action values         ε-greedy · optimistic initial values · UCB · gradient bandit
└── MDPs
    ├── Model-based                 policy evaluation · policy iteration · value iteration · MCTS (AlphaZero-like)
    └── Model-free
        ├── Tabular                 Monte Carlo control · TD(0) · Sarsa · Expected Sarsa · Q-learning
        └── Function approximation  deep Q-learning · REINFORCE · actor–critic
```

| Tab | Book | Environment | What you can see |
|---|---|---|---|
| ε-greedy, optimistic values, UCB, gradient bandit | §2.2–2.8 | 10-armed testbed (stationary or random-walk q*, optional +4 shift) | Average reward and % optimal action over up to 2,000 runs against a comparison method; estimates vs truth; pulls per arm; the Figure 2.6 parameter study |
| Policy evaluation, policy iteration, value iteration | §4.1–4.4 | Editable gridworld | Values and arrows per sweep; Δ per sweep (log scale); one Bellman backup for any hovered cell; in-place vs two-array sweeps; modified policy iteration |
| MCTS · AlphaZero-like | §8.11, §16.6 | Editable gridworld | UCT with random rollouts, or PUCT with a value estimate and prior; where each search looked, root visit shares and Q̄ vs q* |
| Monte Carlo, Sarsa, Expected Sarsa, Q-learning | §5.4, §6.4–6.6 | Editable gridworld | Q(s, a) triangles, greedy policy, visits; return per episode (Figure 6.4) and episodes vs steps (Figure 6.3), with another method trained alongside for comparison |
| TD(0) | §6.1–6.2 | Five-state random walk (Example 6.2) | TD(0) vs constant-α MC estimates against the true values; RMS error averaged over 100 runs for the book's step sizes and yours |
| Deep Q-learning, REINFORCE, actor–critic | §16.5, §13.3–13.5 | Cart-pole (Example 3.4) | Live training episode or the greedy policy in real time; episode lengths; the network's policy and value over pole angle × angular velocity |

### Gridworlds

Presets: cliff walking (Example 6.6), windy gridworld (Example 6.5), the Dyna maze (Example 8.1), the 4 × 4 gridworld (Example 4.1)
and a two-room layout with a pit. Paint walls, goals, cliffs and the start directly on the grid, and set γ and a slip probability
(the chosen action is replaced by a random one), which makes transitions stochastic for the DP methods.
Each step costs −1, a cliff costs −100 and sends you back to the start, and a goal ends the episode.

## Implementation notes

- **Bandits** use the book's testbed: q*(a) ~ N(0, 1), rewards ~ N(q*(a), 1); α = 0 means sample averages.
- **DP** sweeps stop at Δ < θ. With γ = 1 a policy that never reaches a goal has infinite cost; the app detects this and says so.
- **MCTS** normalises Q̄ with the min/max seen in the tree so that `c` does not depend on the reward scale. The AlphaZero-style variant
  has no trained network: its "value network" is v* plus noise of a chosen size, and its prior is a soft-max over one-step look-ahead.
- **Monte Carlo** control is on-policy first-visit with an ε-soft policy (§5.4); episodes are capped at 2,000 steps.
- **Neural networks** are a small hand-written MLP (ReLU, Adam) in plain JavaScript.
  - *DQN* uses experience replay, a soft-updated target network (τ = 0.005), Huber loss and optional Double DQN. It learns from the
    alternative cart-pole reward in Example 3.4 (−1 on failure, 0 otherwise), which keeps Q in [−1, 0]; with +1 per step the
    targets sit near 100 and training was much less stable.
  - *REINFORCE* follows §13.4 (learned baseline, γᵗ factor, both switchable).
  - *Actor–critic* uses n-step returns (n = 32 by default). With n = 1, the book's one-step version, the critic on cart-pole rarely
    sees a failure 20 ms ahead, its values blow up and the policy collapses. Set n = 1 to watch that happen.
  - The 500-step time limit is treated as a time-out, not a terminal state, so values still bootstrap there.

## Running locally

No build step. Serve the folder with any static server:

```sh
python3 -m http.server 8000
# open http://localhost:8000
```

Run the checks (about 20 seconds):

```sh
node tests/check.js
```

They reproduce results from the book: v<sub>π</sub> of the random policy in Figure 4.1, policy iteration and value iteration agreeing on v*,
the 15-step optimum of the windy gridworld, Q-learning's edge path versus Sarsa's safe path on the cliff, TD(0) beating MC on the random walk,
the orderings in Figures 2.2–2.5, MCTS choosing an optimal first move, backprop against finite differences, and all three cart-pole learners
balancing the pole (fixed seeds).

## Deploying to GitHub Pages

The workflow in `.github/workflows/pages.yml` runs the checks and deploys on every push to `main` (or manually via *Run workflow*).
One-time setup: in the repository go to **Settings → Pages → Build and deployment → Source** and choose **GitHub Actions**.

## Layout

| File | Purpose |
|---|---|
| `js/rl.js` | Environments (bandits, gridworld, random walk, cart-pole), all algorithms, MLP with Adam. No DOM, runs in Node |
| `js/plot.js` | Small canvas plotter (lines, bars, markers, log axes, hover) |
| `js/draw.js` | Gridworld, cart-pole, heatmap and random-walk renderers |
| `js/ui.js` | Controls, tab layout, run loop helper, gridworld editor |
| `js/views/*.js` | One file per family of tabs |
| `js/app.js` | Tabs, routing and the animation loop |
| `tests/check.js` | Node sanity checks |

Formulas are typeset with [KaTeX](https://katex.org/) from a CDN; if it can't load, the raw TeX is shown instead.
