// Tabs, routing and the shared animation loop.
(function () {
  'use strict';

  const { UI } = window;

  const GROUPS = [
    { key: 'bandits', name: 'Bandits', sub: 'action values' },
    { key: 'model', name: 'Model-based', sub: 'DP · planning' },
    { key: 'tabular', name: 'Model-free', sub: 'tabular' },
    { key: 'approx', name: 'Model-free', sub: 'function approximation' },
  ];

  const TABS = [
    { id: 'epsilon-greedy', name: 'ε-greedy', group: 'bandits', view: 'bandit', method: 'egreedy' },
    { id: 'optimistic', name: 'Optimistic values', group: 'bandits', view: 'bandit', method: 'optimistic' },
    { id: 'ucb', name: 'UCB', group: 'bandits', view: 'bandit', method: 'ucb' },
    { id: 'gradient-bandit', name: 'Gradient bandit', group: 'bandits', view: 'bandit', method: 'gradient' },
    { id: 'policy-evaluation', name: 'Policy evaluation', group: 'model', view: 'dp', method: 'evaluation' },
    { id: 'policy-iteration', name: 'Policy iteration', group: 'model', view: 'dp', method: 'policy' },
    { id: 'value-iteration', name: 'Value iteration', group: 'model', view: 'dp', method: 'value' },
    { id: 'mcts', name: 'MCTS · AlphaZero-like', group: 'model', view: 'mcts' },
    { id: 'monte-carlo', name: 'Monte Carlo', group: 'tabular', view: 'tabular', method: 'mc' },
    { id: 'td-prediction', name: 'TD(0)', group: 'tabular', view: 'td' },
    { id: 'sarsa', name: 'Sarsa', group: 'tabular', view: 'tabular', method: 'sarsa' },
    { id: 'expected-sarsa', name: 'Expected Sarsa', group: 'tabular', view: 'tabular', method: 'expected' },
    { id: 'q-learning', name: 'Q-learning', group: 'tabular', view: 'tabular', method: 'qlearning' },
    { id: 'dqn', name: 'Deep Q-learning', group: 'approx', view: 'deep', method: 'dqn' },
    { id: 'reinforce', name: 'REINFORCE', group: 'approx', view: 'deep', method: 'reinforce' },
    { id: 'actor-critic', name: 'Actor–critic', group: 'approx', view: 'deep', method: 'ac' },
  ];
  for (const t of TABS) {
    const g = GROUPS.find((x) => x.key === t.group);
    t.groupName = `${g.name} · ${g.sub}`;
  }

  // ---------------------------------------------------------------------------
  // Tab bar
  // ---------------------------------------------------------------------------

  const nav = UI.$('algo-tabs');
  const buttons = new Map();
  for (const g of GROUPS) {
    const tabs = TABS.filter((t) => t.group === g.key);
    const row = UI.el('div', 'tab-group');
    const label = UI.el('span', 'label');
    label.innerHTML = `${g.name} · ${g.sub} <span>· ${tabs.length}</span>`;
    const list = UI.el('div', 'tab-row');
    list.setAttribute('role', 'tablist');
    list.setAttribute('aria-label', `${g.name}, ${g.sub}`);
    for (const t of tabs) {
      const b = UI.el('button', null, t.name);
      b.type = 'button';
      b.setAttribute('role', 'tab');
      b.setAttribute('aria-selected', 'false');
      b.addEventListener('click', () => {
        if (location.hash !== `#${t.id}`) history.pushState(null, '', `#${t.id}`);
        show(t.id);
      });
      list.append(b);
      buttons.set(t.id, b);
    }
    row.append(label, list);
    nav.append(row);
  }

  // ---------------------------------------------------------------------------
  // Views
  // ---------------------------------------------------------------------------

  const container = UI.$('view');
  let current = null;
  let currentId = null;

  function show(id) {
    const tab = TABS.find((t) => t.id === id) || TABS[0];
    if (tab.id === currentId) return;
    if (current && current.destroy) current.destroy();
    currentId = tab.id;
    for (const [tid, b] of buttons) b.setAttribute('aria-selected', String(tid === tab.id));
    const b = buttons.get(tab.id);
    if (b && b.scrollIntoView && b.parentElement.scrollWidth > b.parentElement.clientWidth) {
      b.parentElement.scrollLeft = b.offsetLeft - b.parentElement.offsetLeft - 24;
    }
    UI.$('top-algo').textContent = tab.name;
    document.title = `${tab.name} · Reinforcement Learning Playground`;
    current = window.Views[tab.view](container, tab);
  }

  function route() {
    const id = location.hash.replace(/^#\/?/, '');
    show(TABS.some((t) => t.id === id) ? id : TABS[0].id);
  }
  window.addEventListener('hashchange', route);
  window.addEventListener('popstate', route);

  // One animation loop for whichever view is showing.
  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (current && current.frame) current.frame(dt);
    requestAnimationFrame(frame);
  }

  UI.initTheme();
  UI.onTheme(() => current && current.redraw && current.redraw());
  // KaTeX loads asynchronously (defer); re-typeset once it is there.
  window.addEventListener('load', () => current && current.retex && current.retex());

  route();
  requestAnimationFrame(frame);
})();
