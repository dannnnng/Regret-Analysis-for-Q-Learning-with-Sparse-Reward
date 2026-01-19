# Maze Reinforcement Learning Experiments (UCB + Sparse Reward + ε-greedy)

This script implements a deterministic maze environment and compares two RL approaches:
- UCB-Hoeffding sparse-reward-aware Q-learning (UCB / UCB-H variants)
- ε-greedy Q-learning baseline

The main program runs multiple experiments on two fixed maze seeds and saves reward curves, path visualizations, Q-tables, and summary statistics.

## Files
- `QLearningUCBsparse-Maze.py`: main script with environment, algorithms, training, and visualization.

## Environment and Maze
### State and Action
- State: `(row, col)` on a `size * size` grid.
- Actions: 5 discrete actions.
  - `UP` `RIGHT` `DOWN` `LEFT` `STAY`
  - `STAY` means no movement.

### Transition and Termination
- Hitting a wall or boundary: state does not change and is counted as a wall hit.
- Reaching `goal`: counted as success and rewarded.
- Each episode ends after at most `horizon` steps.

### Reward Design
- Reach goal: `+1.0`
- Wall hit: `wall_penalty` (default `-100.0`)
- Stay in place: `stay_penalty` (default `-0.2`)
- Normal move: `move_penalty` (default `-0.2`)

### Maze Generation
- `build_corridor_maze()` uses DFS backtracking to generate corridor-style mazes, keeps the start neighborhood open, and opens cells near the goal.
- The main flow always uses this generator; `MazeEnv._generate_maze()` is only used when no external grid is provided.

## Algorithms
### 1) QLearning UCB Hoeffding Sparse (UCB with sparse reward awareness)
- Goal: improve exploration in sparse reward settings.
- Q initialization: `Q(s,a) = s = sparse_fraction * H`.
- V initialization: `0.0`.
- Visit counts: `N(s,a)`.
- UCB bonus:
  - `ι = log(S * A * H * K / failure_prob)`
  - `δ = bonus_constant = 1`
- Update (per step):
  - `α = (H + 1) / (H + t)`
  - `Q(s,a) <- (1 - α) * Q(s,a) + α * (r + V(s') + bonus)`
  - `V(s) <- min(s, max_a Q(s,a))`
- Evaluation: greedy policy rollout; report success rate and average reward.
- Extra: custom random pools (episode/step) for reproducible tie-breaking and sampling.

#### Parameter Meaning and Rationale
- `sparse_fraction`: defines the sparse-reward horizon proxy `s = sparse_fraction * H`. It sets the optimistic prior scale for Q-values and the exploration bonus magnitude. `0.01` models a highly sparse reward regime; `1.0` (UCB-H) removes sparsity assumptions and makes the bonus effectively scale with the full horizon.
- `failure_prob`: appears in `ι = log(S * A * H * K / failure_prob)` as a confidence parameter. Smaller values increase the log term, producing a larger bonus (more exploration) and more conservative confidence; larger values reduce exploration pressure.
- `bonus_constant`: a multiplicative factor on the exploration bonus. `1.0` is a neutral baseline; increasing it strengthens exploration and can slow exploitation, decreasing it does the opposite.

#### Design Notes
- `c` is computed as `bonus_constant / (s * sqrt(H * ι))` so that the resulting bonus `c * s * sqrt(H * ι / t)` stays bounded and roughly on the same scale across different horizons and sparsity settings. This keeps the bonus from dominating rewards when `H` is large.
- The learning-rate schedule `α = (H + 1) / (H + t)` is a decreasing step size commonly used in finite-horizon tabular settings to temper updates as visit count `t` grows. It emphasizes early optimistic updates while stabilizing later.
- Q is initialized to `s = sparse_fraction * H` to encode optimistic values in sparse-reward tasks, encouraging exploration before enough positive rewards are observed. Larger initialization yields more optimistic exploration; smaller values reduce that effect.

### 2) QLearning εGreedy (baseline)
- Q-learning update:
  - `Q(s,a) <- Q(s,a) + 0.2 * (r + 0.99 * max_a' Q(s',a') - Q(s,a))`
- Behavior policy: fixed ε-greedy (ε = 0.1 , no decay).
- Evaluation: greedy policy (`argmax_a Q(s,a)`) after training.

## Experiment Flow (main)
The main function runs following experiments:
- Maze seeds `1` and `12`
- Algorithms and settings:
  - `UCB-H` (`sparse_aware=0.01`)
  - `UCB-H` (`sparse_fraction=1.0`)
- `ε-greedy` (baseline)

Each run outputs reward curves, Q tables, path visualizations, and config data, plus a three-method comparison plot on the same maze.

## Experiment Design Notes
- Maze seeds `1` and `12` provide two deterministic but different layouts for a small cross-check. They are not special beyond producing distinct maze structures.
- `episodes=200` is a trade-off between runtime and learning progress for this grid size. Convergence is typically assessed by stabilization of reward curves and success rate; increase episodes if curves keep trending upward.
- `eval_episodes=200` reduces variance in success-rate estimates by averaging over more rollouts. Fewer episodes yield noisier estimates; more episodes reduce noise but increase runtime.

## How to Run
```bash
python QLearningUCBsparse-Maze.py
```

### Command-line Arguments
- `--seed`: base random seed (default: random)
- `--output_dir`: output directory (default: timestamp folder)
- `--wall_penalty`: wall-hit penalty
- `--stay_penalty`: stay penalty
- `--move_penalty`: move penalty
- `--episodes`: number of training episodes (default: 200)
- `--horizon`: max steps per episode (default: 2000)

### Other Fixed Parameters (in `main()`)
- `maze_size=15`
- `failure_prob=0.1`
- `bonus_constant=1.0`
- `sparse_fraction=0.01`
- `log_interval=100`
- `eval_episodes=200`
- `record_paths=True`

## Output Structure
Default output is a timestamped folder (for example `20260113_185648/`), with one subfolder per maze seed:
```
<output_dir>/
  maze_seed_1/
    plots/
    visualizations/
    animations/
    data/
    tables/
  maze_seed_12/
    ...
```

### Typical Outputs
- `plots/reward_vs_step_*.png`: reward curves
- `plots/reward_vs_step_compare_three.png`: three-method comparison curve
- `visualizations/maze_layout.png`: maze layout
- `visualizations/q_greedy_path_*.png`: greedy path from Q-table
- `visualizations/q_greedy_path_compare_three.png`: path comparison
- `animations/episode_paths_*.gif`: training path animation (when recording)
- `tables/q_table_detailed_*.txt`: full Q table with coordinates
- `tables/q_table_summary_*.txt`: Q table summary stats
- `tables/visit_counts_*.txt`: visit counts
- `data/experiment_config_*.json`: experiment config and results

## Dependencies
- Python 3.9+ (3.10+ recommended)
- `numpy`
- `matplotlib`
- `pillow` (for GIF export)

## Reproducibility
- Use `seed` to fix the base random seed.
- UCB uses custom random pools for reproducible tie-breaking and evaluation.
- Maze generation uses fixed `maze_seed`; different seeds produce different layouts.

## Notes
- The `STAY` action is included to study the effect of no-move behavior in sparse reward tasks and has a separate penalty.
- To run only a subset of algorithms, edit the `experiments` list in `main()`.
