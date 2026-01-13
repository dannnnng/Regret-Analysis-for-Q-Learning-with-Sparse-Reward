from __future__ import annotations
import math
import random
import time
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import animation

Coordinate = Tuple[int, int]


class MazeEnv:
    """Maze with deterministic transitions and terminal rewards."""

    # Five actions: up/down/left/right + stay in place.
    ACTIONS: Sequence[Coordinate] = (
        (-1, 0),  # up
        (0, 1),  # right
        (1, 0),  # down
        (0, -1),  # left
        (0, 0),  # stay (no movement)
    )

    ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT", "STAY"]

    def __init__(self, size: int = 15, horizon: int = 500, seed: int = 42,
                 grid: np.ndarray | None = None, start: Coordinate | None = None,
                 goal: Coordinate | None = None, wall_penalty: float = -1,
                 stay_penalty: float = -0.5,
                 move_penalty: float = -0.2) -> None:
        self.size = size
        self.horizon = horizon
        # Fixed seed for maze generation to keep the environment reproducible.
        self.random = random.Random(seed)
        self.start: Coordinate = start if start is not None else (0, 0)
        self.goal: Coordinate = goal if goal is not None else (size - 1, size - 1)
        self.grid = self._generate_maze(seed) if grid is None else grid.copy()
        self.state: Coordinate = self.start
        self.step_count = 0
        self.wall_penalty = wall_penalty  # Penalty for hitting a wall.
        self.stay_penalty = stay_penalty  # Penalty for staying in place.
        self.move_penalty = move_penalty  # Penalty for a normal move.
        self.seed = seed  # Stored seed.

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return len(self.ACTIONS)

    def _generate_maze(self, seed: int) -> np.ndarray:
        """Creates a maze with multiple carved corridors plus a guaranteed start-goal path."""
        # Default maze generator used only when no external grid is provided.
        rng = random.Random(seed)
        grid = np.ones((self.size, self.size), dtype=np.int8)
        safe_path: set[Coordinate] = set()

        def carve_path(a: Coordinate, b: Coordinate) -> None:
            # Carve a Manhattan path between two points (horizontal then vertical).
            r, c = a
            safe_path.add((r, c))
            while c != b[1]:
                c += 1 if b[1] > c else -1
                safe_path.add((r, c))
            while r != b[0]:
                r += 1 if b[0] > r else -1
                safe_path.add((r, c))

        # Guarantee connections between key waypoints to avoid disconnected maps.
        carve_path((0, 0), (0, self.size - 1))
        carve_path((0, self.size - 1), (self.size - 1, self.size - 1))
        carve_path(self.start, self.goal)
        carve_path((0, 0), self.start)

        # Add an always-open diagonal corridor to diversify routes.
        for idx in range(self.size):
            safe_path.add((idx, idx))

        # Open safe_path cells, plus random openings elsewhere to create branches.
        for r in range(self.size):
            for c in range(self.size):
                if (r, c) in safe_path or (r, c) == self.goal:
                    grid[r, c] = 0
                elif rng.random() < 0.3:
                    grid[r, c] = 0
        return grid

    def reset(self) -> Coordinate:
        self.state = self.start
        self.step_count = 0
        return self.state

    def step(self, action: int) -> Tuple[Coordinate, float, bool, dict]:
        reward = 0.0
        hit_wall = False
        is_stay = False

        # Try to move.
        dr, dc = self.ACTIONS[action]
        target_r = self.state[0] + dr
        target_c = self.state[1] + dc

        next_coord = self.state
        # Check whether the action is staying in place.
        if dr == 0 and dc == 0:
            is_stay = True
            # Stay in place; state does not change.
        else:
            # Check for out-of-bounds moves.
            if not (0 <= target_r < self.size and 0 <= target_c < self.size):
                hit_wall = True
            else:
                # Check for walls.
                if self.grid[target_r, target_c] == 0:
                    next_coord = (target_r, target_c)
                    self.state = next_coord
                else:
                    hit_wall = True

        self.step_count += 1
        reached_goal = next_coord == self.goal
        done = self.step_count >= self.horizon

        if reached_goal:
            reward = 1.0
        elif hit_wall:
            reward = self.wall_penalty
        elif is_stay:
            reward = self.stay_penalty
        else:
            reward = self.move_penalty

        return next_coord, reward, done, {"hit_wall": hit_wall, "is_stay": is_stay}

    def coord_to_state(self, coord: Coordinate) -> int:
        return coord[0] * self.size + coord[1]

    def state_to_coord(self, state: int) -> Coordinate:
        return divmod(state, self.size)

    def render(self, path: Sequence[Coordinate] | None = None) -> str:
        path_set = set(path) if path else set()
        rows: List[str] = []
        for r in range(self.size):
            chars: List[str] = []
            for c in range(self.size):
                coord = (r, c)
                if coord == self.start:
                    ch = 'S'
                elif coord == self.goal:
                    ch = 'G'
                elif self.grid[r, c] == 1:
                    ch = '#'
                else:
                    ch = '.'
                if coord in path_set and ch == '.':
                    ch = '*'
                chars.append(ch)
            rows.append(''.join(chars))
        return '\n'.join(rows)


def build_corridor_maze(size: int = 15, seed: int = 7) -> np.ndarray:
    """Generates a maze with explicit walls using a DFS-backtracking algorithm."""
    # Corridor-style maze generator used by the main experiment for consistency.
    # Fixed seed for maze generation to keep layouts reproducible.
    rng = random.Random(seed)
    grid = np.ones((size, size), dtype=np.int8)

    def carve_passage(cell: Coordinate, nxt: Coordinate) -> None:
        # Carve a passage by opening both cells and the wall between them.
        r1, c1 = cell
        r2, c2 = nxt
        grid[r1, c1] = 0
        grid[r2, c2] = 0
        grid[(r1 + r2) // 2, (c1 + c2) // 2] = 0

    start = (0, 0)
    stack: List[Coordinate] = [start]
    visited = {start}

    def cell_neighbors(coord: Coordinate) -> List[Coordinate]:
        # Neighbor cells are 2 steps away to preserve walls between corridors.
        r, c = coord
        dirs = [(-2, 0), (0, 2), (2, 0), (0, -2)]
        result: List[Coordinate] = []
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                result.append((nr, nc))
        rng.shuffle(result)
        return result

    grid[start] = 0
    while stack:
        current = stack[-1]
        neighbors = [nbr for nbr in cell_neighbors(current) if nbr not in visited]
        if not neighbors:
            stack.pop()
            continue
        nxt = neighbors[0]
        carve_passage(current, nxt)
        visited.add(nxt)
        stack.append(nxt)

    # Ensure the goal region is open and locally connected.
    grid[size - 1, size - 1] = 0
    grid[size - 2, size - 1] = 0
    grid[size - 1, size - 2] = 0

    # Add a limited number of random shortcuts to create loops.
    for _ in range(size):
        r = rng.randrange(1, size - 1)
        c = rng.randrange(1, size - 1)
        grid[r, c] = 0

    # Ensure cells adjacent to the start are free so the agent can move initially.
    for dr, dc in [(1, 0), (0, 1)]:
        nr, nc = dr, dc
        if 0 <= nr < size and 0 <= nc < size:
            grid[nr, nc] = 0

    return grid


@dataclass
class TrainingStats:
    episodes: List[int]
    rewards: List[float]
    successes: List[bool]  # Whether each episode reached the goal.
    success_rate: float
    episode_paths: List[List[Coordinate]] | None = None


def plot_reward_trace(episodes: Sequence[int], rewards: Sequence[float],
                      out_path: str = "reward_vs_step.png", window: int = 50) -> str:
    """Plots reward vs. step (episode index) and saves to disk."""
    plt.figure(figsize=(10, 4))
    plt.plot(episodes, rewards, label="Episode Reward", alpha=0.3)
    plt.xlabel("Episode", fontsize=14, fontweight="bold")
    plt.ylabel("Reward", fontsize=14, fontweight="bold")
    ax = plt.gca()
    ax.yaxis.set_label_coords(-0.08, 0.45)
    # No title per request.
    plt.xlim(0, 200)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="lower right", fontsize=18)
    ax.set_frame_on(True)
    ax.spines["left"].set_bounds(*ax.get_ylim())
    ax.spines["bottom"].set_bounds(*ax.get_xlim())
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    plt.tight_layout()
    # Ensure the directory exists.
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_reward_comparison_three(stats_a: TrainingStats, label_a: str,
                                 stats_b: TrainingStats, label_b: str,
                                 stats_c: TrainingStats, label_c: str,
                                 out_path: str = "reward_vs_step_compare_three.png") -> str:
    """Plots reward vs. episode for three runs on the same figure."""
    plt.figure(figsize=(10, 4))
    plt.plot(stats_a.episodes, stats_a.rewards, color="red", alpha=0.6, label=label_a)
    plt.plot(stats_b.episodes, stats_b.rewards, color="black", alpha=0.6, label=label_b)
    plt.plot(stats_c.episodes, stats_c.rewards, color="blue", alpha=0.6, label=label_c)
    plt.xlabel("Episode", fontsize=14, fontweight="bold")
    plt.ylabel("Reward", fontsize=14, fontweight="bold")
    ax = plt.gca()
    ax.yaxis.set_label_coords(-0.08, 0.45)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(0, 200)
    plt.legend(loc="lower right", fontsize=18)
    ax.set_frame_on(True)
    ax.spines["left"].set_bounds(*ax.get_ylim())
    ax.spines["bottom"].set_bounds(*ax.get_xlim())
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def save_maze_visualization(grid: np.ndarray, start: Coordinate, goal: Coordinate,
                            path: Sequence[Coordinate] | None = None,
                            out_path: str = "maze_layout.png",
                            marker_style: str = "line") -> str:
    """Saves a color visualization of the maze with optional path overlay."""
    # Ensure the directory exists.
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = colors.ListedColormap(["white", "black"])
    ax.imshow(grid, cmap=cmap, origin="upper")
    ax.scatter(start[1], start[0], c="green", marker="o", s=60, label="Start")
    ax.scatter(goal[1], goal[0], c="red", marker="*", s=80, label="Goal")
    if path:
        rows = [coord[0] for coord in path]
        cols = [coord[1] for coord in path]
        if marker_style == "hollow_blue":
            ax.plot(cols, rows, linestyle="None", marker="o", markersize=3,
                    markerfacecolor="none", markeredgecolor="blue",
                    color="blue", label="Path")
        elif marker_style == "black_dashed":
            ax.plot(cols, rows, color="black", linewidth=1.2, linestyle="--",
                    marker="^", markersize=3, markerfacecolor="none",
                    markeredgecolor="black", markevery=2, label="Path")
        else:
            ax.plot(cols, rows, color="blue", linewidth=1.5, label="Path")
    ax.set_xticks([])
    ax.set_yticks([])
    # No title per request.
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_maze_visualization_compare_three(grid: np.ndarray, start: Coordinate, goal: Coordinate,
                                          path_a: Sequence[Coordinate], label_a: str,
                                          path_b: Sequence[Coordinate], label_b: str,
                                          path_c: Sequence[Coordinate], label_c: str,
                                          out_path: str = "maze_path_compare_three.png") -> str:
    """Saves a comparison visualization for three paths."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = colors.ListedColormap(["white", "black"])
    ax.imshow(grid, cmap=cmap, origin="upper")
    ax.scatter(start[1], start[0], c="green", marker="o", s=60, label="Start")
    ax.scatter(goal[1], goal[0], c="red", marker="*", s=80, label="Goal")

    if path_a:
        a_rows = [coord[0] for coord in path_a]
        a_cols = [coord[1] for coord in path_a]
        ax.plot(a_cols, a_rows, color="red", linewidth=1.5, linestyle="--",
                label=label_a)

    if path_b:
        b_rows = [coord[0] for coord in path_b]
        b_cols = [coord[1] for coord in path_b]
        ax.plot(b_cols, b_rows, color="black", linewidth=1.2,
                label=label_b)

    if path_c:
        c_rows = [coord[0] for coord in path_c]
        c_cols = [coord[1] for coord in path_c]
        ax.plot(c_cols, c_rows, linestyle="None", marker="o", markersize=3,
                markerfacecolor="none", markeredgecolor="blue",
                color="blue", label=label_c)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_episode_animation(grid: np.ndarray, start: Coordinate, goal: Coordinate,
                           episode_paths: Sequence[Sequence[Coordinate]],
                           rewards: Sequence[float] | None = None,
                           out_path: str = "episode_paths.gif",
                           interval_ms: int = 120) -> str:
    """Saves an animation that shows each episode's path in sequence."""
    # Ensure the directory exists.
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = colors.ListedColormap(["white", "black"])
    ax.imshow(grid, cmap=cmap, origin="upper")
    ax.scatter(start[1], start[0], c="green", marker="o", s=60, label="Start")
    ax.scatter(goal[1], goal[0], c="red", marker="*", s=80, label="Goal")
    path_line, = ax.plot([], [], color="blue", linewidth=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Episode Paths")
    ax.legend(loc="upper right")

    def init() -> tuple:
        path_line.set_data([], [])
        return (path_line,)

    def update(frame_idx: int) -> tuple:
        path = episode_paths[frame_idx]
        rows = [coord[0] for coord in path]
        cols = [coord[1] for coord in path]
        path_line.set_data(cols, rows)
        if rewards is not None and frame_idx < len(rewards):
            ax.set_title(f"Episode {frame_idx + 1} | reward={rewards[frame_idx]:.2f}")
        else:
            ax.set_title(f"Episode {frame_idx + 1}")
        return (path_line,)

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(episode_paths),
        interval=interval_ms,
        blit=True,
        repeat=False,
    )
    anim.save(out_path, writer="pillow")
    plt.close(fig)
    return out_path


class RandomNumberPool:
    """Large random number pool, pre-generated and consumed sequentially."""

    def __init__(self, pool_size: int = 1000000, seed: int = 42):
        """Initialize the random pool.

        Args:
            pool_size: Size of the random pool.
            seed: Seed for the random generator.
        """
        self.pool_size = pool_size
        self.pool = None
        self.index = 0
        self.seed = seed
        self._generate_pool()

    def _generate_pool(self):
        """Generate the random pool."""
        np.random.seed(self.seed)
        # Draw uniform random numbers in [0, 1).
        self.pool = np.random.random(self.pool_size)

    def get_random(self) -> float:
        """Get one random value from the pool."""
        if self.index >= self.pool_size:
            # Refill when the pool is exhausted.
            self._refill_pool()

        value = self.pool[self.index]
        self.index += 1
        return value

    def get_random_int(self, low: int, high: int) -> int:
        """Get a random integer in [low, high).

        Args:
            low: Inclusive lower bound.
            high: Exclusive upper bound.
        """
        rand_float = self.get_random()
        return int(low + rand_float * (high - low))

    def get_random_choice(self, array):
        """Randomly choose one element from an array."""
        if len(array) == 0:
            return None
        idx = self.get_random_int(0, len(array))
        return array[idx]

    def _refill_pool(self):
        """Refill the random pool."""
        # Use the current index as a new seed to vary refills.
        new_seed = int(self.seed + self.index)
        np.random.seed(new_seed)
        self.pool = np.random.random(self.pool_size)
        self.index = 0

    def reset(self):
        """Reset the index back to the start."""
        self.index = 0


class EpisodeRandomPool:
    """Random pools dedicated to each episode."""

    def __init__(self, base_seed: int, pool_size: int = 10000):
        """Initialize the per-episode random pools.

        Args:
            base_seed: Base seed.
            pool_size: Pool size per episode.
        """
        self.base_seed = base_seed
        self.pool_size = pool_size
        self.episode_pools = {}

    def get_episode_pool(self, episode_idx: int) -> RandomNumberPool:
        """Get or create the random pool for a specific episode."""
        if episode_idx not in self.episode_pools:
            # Use episode index to create a unique seed.
            episode_seed = self.base_seed + episode_idx * 1000
            self.episode_pools[episode_idx] = RandomNumberPool(
                pool_size=self.pool_size,
                seed=episode_seed
            )
        return self.episode_pools[episode_idx]

    def get_step_pool(self, episode_idx: int, step_idx: int) -> RandomNumberPool:
        """Get or create the random pool for a specific episode and step."""
        pool_key = (episode_idx, step_idx)
        if pool_key not in self.episode_pools:
            # Use episode and step indices to create a unique seed.
            step_seed = self.base_seed + episode_idx * 10000 + step_idx * 100
            self.episode_pools[pool_key] = RandomNumberPool(
                pool_size=1000,  # Step pool can be smaller.
                seed=step_seed
            )
        return self.episode_pools[pool_key]


class QLearningUCBHoeffdingSparse:
    """Algorithm 1 implementation with sparse reward awareness."""

    def __init__(self, env: MazeEnv, horizon: int, episodes: int,
                 failure_prob: float = 0.1, bonus_constant: float = 1.0,
                 sparse_fraction: float = 0.1, seed: Optional[int] = None) -> None:
        self.env = env
        self.H = horizon
        self.K = episodes
        self.S = env.n_states
        self.A = env.n_actions
        self.sparse_fraction = sparse_fraction
        self.sparse_steps = self.sparse_fraction * self.H  # s in sparse reward description.
        self.failure_prob = failure_prob
        # Set the base random seed.
        if seed is None:
            seed = random.randint(0, 1000000)
        self.base_seed = seed

        # Initialize the random pool system.
        self.episode_random_pool = EpisodeRandomPool(base_seed=seed, pool_size=10000)
        self.eval_random_pool = RandomNumberPool(pool_size=50000, seed=seed + 1000000)
        self.global_random_pool = RandomNumberPool(pool_size=100000, seed=seed + 2000000)

        total_steps = self.H * self.K
        self.iota = math.log(max(self.S * self.A * total_steps / self.failure_prob, 1.0001))
        # Scale c so that b_t = c * s * sqrt(H iota / t) stays well below 1.
        self.c = bonus_constant / (self.sparse_steps * math.sqrt(self.H * self.iota))

        # Initialize all Q-values to the sparse-step scale s.
        self.q_init_value = self.sparse_steps

        # Q values initialized to s.
        self.Q = np.full((self.S, self.A), self.q_init_value, dtype=float)

        # V values initialized to 0.
        self.V = np.zeros(self.S, dtype=float)

        self.N = np.zeros((self.S, self.A), dtype=int)

        # Record initial Q statistics.
        self.initial_q_min = np.min(self.Q)
        self.initial_q_max = np.max(self.Q)
        self.initial_q_mean = np.mean(self.Q)
        self.initial_q_std = np.std(self.Q)

        # Record initial V statistics.
        self.initial_v_min = np.min(self.V)
        self.initial_v_max = np.max(self.V)
        self.initial_v_mean = np.mean(self.V)
        self.initial_v_std = np.std(self.V)

    def train(self, log_interval: int = 50, record_paths: bool = False) -> TrainingStats:
        rewards: List[float] = []
        successes: List[bool] = []  # Whether each episode reached the goal.
        episodes_axis: List[int] = []
        episode_paths: List[List[Coordinate]] | None = [] if record_paths else None

        # Track exploration statistics.
        exploration_stats = {
            'unique_state_actions': []  # Unique state-action pairs per episode.
        }

        for episode in range(1, self.K + 1):
            # Get the random pool for this episode.
            episode_pool = self.episode_random_pool.get_episode_pool(episode - 1)  # Zero-based index.

            state_coord = self.env.reset()
            if episode_paths is not None:
                episode_coords: List[Coordinate] = [state_coord]
            state = self.env.coord_to_state(state_coord)
            episode_total_reward = 0.0
            episode_success = False  # Track whether this episode reached the goal.

            # Track exploration stats for this episode.
            episode_unique_sa = set()

            for h in range(self.H):
                # Get the random pool for this step.
                step_pool = self.episode_random_pool.get_step_pool(episode - 1, h)

                action = self._greedy_action_with_pool(state, step_pool)

                episode_unique_sa.add((state, action))

                next_coord, reward, done, info = self.env.step(action)

                if episode_paths is not None:
                    episode_coords.append(next_coord)
                next_state = self.env.coord_to_state(next_coord)
                t = self.N[state, action] + 1
                self.N[state, action] = t

                # Accumulate episode reward.
                episode_total_reward += reward

                # Check whether the goal is reached.
                if next_coord == self.env.goal:
                    episode_success = True

                alpha = (self.H + 1) / (self.H + t)
                bonus = self.c * self.sparse_steps * math.sqrt((self.H * self.iota) / max(t, 1))
                next_value = 0.0 if h == self.H - 1 or done else self.V[next_state]
                target = reward + next_value + bonus
                self.Q[state, action] = (1 - alpha) * self.Q[state, action] + alpha * target
                self.V[state] = min(self.sparse_steps, np.max(self.Q[state]))
                state = next_state
                if done:
                    break

            rewards.append(episode_total_reward)
            successes.append(episode_success)  # Record success.
            episodes_axis.append(episode)
            if episode_paths is not None:
                episode_paths.append(episode_coords)

            # Record exploration statistics.
            exploration_stats['unique_state_actions'].append(len(episode_unique_sa))

            if episode % log_interval == 0:
                recent_rewards = rewards[-log_interval:]
                recent_successes = successes[-log_interval:]

                avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
                success_rate = sum(recent_successes) / len(recent_successes) if recent_successes else 0.0

                recent_unique = exploration_stats['unique_state_actions'][-log_interval:]
                avg_unique = sum(recent_unique) / len(recent_unique) if recent_unique else 0

                print(f"Episode {episode:4d}/{self.K}: "
                      f"avg reward={avg_reward:.2f}, "
                      f"success rate={success_rate:.2f}, "
                      f"unique SA={avg_unique:.1f}")

                # Print Q-value statistics.
                if episode % (log_interval * 5) == 0:
                    self._print_value_statistics(episode)

        # Compute the final success rate.
        final_successes = successes[-min(100, len(successes)):]
        success_rate = sum(final_successes) / max(1, len(final_successes))

        # Print final exploration statistics.
        self._print_exploration_summary(exploration_stats)

        return TrainingStats(
            episodes=episodes_axis,
            rewards=rewards,
            successes=successes,
            success_rate=success_rate,
            episode_paths=episode_paths,
        )

    def _print_value_statistics(self, episode: int):
        """Print Q and V statistics."""
        negative_q_count = np.sum(self.Q < 0)
        positive_q_count = np.sum(self.Q > 0)

        print(f"  Value-statistics at episode {episode}:")
        print(f"    Negative Q-values: {negative_q_count} ({negative_q_count / (self.S * self.A) * 100:.1f}%)")
        print(f"    Positive Q-values: {positive_q_count} ({positive_q_count / (self.S * self.A) * 100:.1f}%)")
        print(f"    Q min/max: {np.min(self.Q):.2f}/{np.max(self.Q):.2f}")
        print(f"    Q mean/std: {np.mean(self.Q):.2f}/{np.std(self.Q):.2f}")
        print(f"    V min/max: {np.min(self.V):.2f}/{np.max(self.V):.2f}")
        print(f"    V mean/std: {np.mean(self.V):.2f}/{np.std(self.V):.2f}")

    def _print_exploration_summary(self, exploration_stats: dict):
        """Print exploration summary."""
        print("\n" + "=" * 60)
        print("探索统计摘要:")
        print("=" * 60)

        total_unique = sum(exploration_stats['unique_state_actions'])
        avg_unique = total_unique / len(exploration_stats['unique_state_actions'])
        max_unique = max(exploration_stats['unique_state_actions'])

        print(f"平均每episode探索的唯一状态-动作对: {avg_unique:.1f} (最大: {max_unique})")
        print(f"总探索的状态-动作对数量: {total_unique}")
        print(f"总状态-动作对空间大小: {self.S * self.A}")
        print(f"探索覆盖率: {total_unique / (self.S * self.A) * 100:.1f}%")

    def _greedy_action_with_pool(self, state: int, random_pool: RandomNumberPool) -> int:
        """Select a greedy action using a random pool to break ties."""
        q_values = self.Q[state]
        max_value = np.max(q_values)
        max_actions = np.flatnonzero(np.isclose(q_values, max_value))

        # Use the random pool to choose among ties.
        if len(max_actions) == 1:
            return int(max_actions[0])
        else:
            # Random tie-breaking.
            return int(max_actions[random_pool.get_random_int(0, len(max_actions))])

    def greedy_path_from_q(self, env: MazeEnv, deterministic: bool = True,
                           episode_seed: Optional[int] = None) -> List[Coordinate]:
        """Follows the greedy policy derived from the Q-table to get a path."""
        # Create a random pool for evaluation.
        if episode_seed is None:
            # Draw a seed from the global pool.
            episode_seed = int(self.global_random_pool.get_random() * 1000000)
        eval_pool = RandomNumberPool(pool_size=1000, seed=episode_seed)

        coord = env.reset()
        coords = [coord]
        state = env.coord_to_state(coord)

        for h in range(self.H):
            # Create an independent random pool for each step.
            step_pool = RandomNumberPool(pool_size=100, seed=episode_seed + h * 100)

            q_values = self.Q[state]
            max_value = np.max(q_values)
            max_actions = np.flatnonzero(np.isclose(q_values, max_value))

            if deterministic:
                action = int(max_actions[0])
            else:
                # Use the step pool for random tie-breaking.
                action = int(max_actions[step_pool.get_random_int(0, len(max_actions))])

            coord, _, done, _ = env.step(action)
            coords.append(coord)
            state = env.coord_to_state(coord)
            if done:
                break
        return coords

    def rollout(self, env: MazeEnv, episode_seed: Optional[int] = None) -> Tuple[List[Coordinate], float, bool]:
        """Follows the greedy policy for a single episode and returns the positions, total reward, and success status."""
        # Create a random pool for evaluation.
        if episode_seed is None:
            episode_seed = int(self.eval_random_pool.get_random() * 1000000)
        eval_pool = RandomNumberPool(pool_size=1000, seed=episode_seed)

        coord = env.reset()
        coords = [coord]
        state = env.coord_to_state(coord)
        total_reward = 0.0
        success = False  # Track whether the goal is reached.

        for h in range(self.H):
            # Create an independent random pool for each step.
            step_pool = RandomNumberPool(pool_size=100, seed=episode_seed + h * 100)

            q_values = self.Q[state]
            max_value = np.max(q_values)
            max_actions = np.flatnonzero(np.isclose(q_values, max_value))

            # Use the step pool to select an action.
            if len(max_actions) == 1:
                action = int(max_actions[0])
            else:
                action = int(max_actions[step_pool.get_random_int(0, len(max_actions))])

            coord, reward, done, _ = env.step(action)
            coords.append(coord)
            total_reward += reward

            # Check whether the goal is reached.
            if coord == env.goal:
                success = True

            state = env.coord_to_state(coord)
            if done:
                break
        return coords, total_reward, success

    def evaluate(self, env: MazeEnv, episodes: int = 50) -> Tuple[float, float]:
        """Estimates the success rate and average reward of the learned policy."""
        successes = 0
        total_rewards = 0.0

        for eval_episode in range(episodes):
            # Use the evaluation random pool.
            eval_pool = self.eval_random_pool

            coord = env.reset()
            state = env.coord_to_state(coord)
            episode_reward = 0.0
            episode_success = False

            for h in range(self.H):
                # Use the global pool to seed a step pool.
                step_pool = RandomNumberPool(
                    pool_size=100,
                    seed=int(self.global_random_pool.get_random() * 1000000 + h * 100)
                )

                q_values = self.Q[state]
                max_value = np.max(q_values)
                max_actions = np.flatnonzero(np.isclose(q_values, max_value))

                # Use the step pool to select an action.
                if len(max_actions) == 1:
                    action = int(max_actions[0])
                else:
                    action = int(max_actions[step_pool.get_random_int(0, len(max_actions))])

                coord, reward, done, _ = env.step(action)
                episode_reward += reward

                # Check whether the goal is reached.
                if coord == env.goal:
                    episode_success = True

                state = env.coord_to_state(coord)
                if done:
                    if episode_success:
                        successes += 1
                    break

            total_rewards += episode_reward

        success_rate = successes / episodes
        avg_reward = total_rewards / episodes
        return success_rate, avg_reward

    def save_q_table_with_coordinates(self, filepath: str, env: MazeEnv) -> None:
        """Save the Q-table with coordinates and action descriptions."""
        # Ensure the directory exists.
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            # Write header information.
            f.write("# Q-Table with State Coordinates and Action Descriptions\n")
            f.write("# ======================================================\n")
            f.write(f"# Maze Size: {env.size}x{env.size}\n")
            f.write(f"# Number of States: {self.S}\n")
            f.write(f"# Number of Actions: {self.A}\n")
            f.write(f"# Start: {env.start}\n")
            f.write(f"# Goal: {env.goal}\n")
            f.write(f"# Training Episodes: {self.K}\n")
            f.write(f"# Horizon: {self.H}\n")
            f.write(f"# Initial Q Value: {self.q_init_value}\n")
            f.write("# ======================================================\n\n")

            # Write action mapping.
            f.write("# Action Mapping:\n")
            for i, (dr, dc) in enumerate(env.ACTIONS):
                action_name = env.ACTION_NAMES[i] if i < len(env.ACTION_NAMES) else f"ACTION_{i}"
                f.write(f"#   Action {i}: {action_name} (move: {dr}, {dc})\n")
            f.write("\n")

            # Write Q-values for each state.
            f.write("# State Information and Q-Values:\n")
            f.write(
                "# Format: StateID (row, col) | IsStart | IsGoal | IsWall | UP | RIGHT | DOWN | LEFT | STAY | BestAction\n")
            f.write(
                "# ---------------------------------------------------------------------------------------------------\n")

            for state in range(self.S):
                coord = env.state_to_coord(state)
                row, col = coord

                # Determine state type.
                is_start = coord == env.start
                is_goal = coord == env.goal
                is_wall = env.grid[row, col] == 1 if 0 <= row < env.size and 0 <= col < env.size else True

                # Get the best action.
                q_values = self.Q[state]
                best_action_idx = int(np.argmax(q_values))
                best_action_name = env.ACTION_NAMES[best_action_idx] if best_action_idx < len(
                    env.ACTION_NAMES) else f"ACTION_{best_action_idx}"

                # Write state info.
                f.write(f"State {state:3d} ({row:2d}, {col:2d}): ")
                f.write(f"Start={1 if is_start else 0}, ")
                f.write(f"Goal={1 if is_goal else 0}, ")
                f.write(f"Wall={1 if is_wall else 0}")
                f.write(" | ")

                # Write Q-values.
                for action in range(self.A):
                    q_value = self.Q[state, action]
                    f.write(f"{q_value:8.4f} ")

                # Write best action.
                f.write(f"| Best: {best_action_name} (Action {best_action_idx})\n")

            # Write best policy summary.
            f.write("\n# Best Policy Summary:\n")
            f.write("# --------------------\n")

            for state in range(self.S):
                coord = env.state_to_coord(state)
                row, col = coord
                is_wall = env.grid[row, col] == 1 if 0 <= row < env.size and 0 <= col < env.size else True

                if is_wall:
                    continue  # Skip wall states.

                q_values = self.Q[state]
                best_action_idx = int(np.argmax(q_values))
                best_action_name = env.ACTION_NAMES[best_action_idx] if best_action_idx < len(
                    env.ACTION_NAMES) else f"ACTION_{best_action_idx}"
                f.write(f"  ({row:2d}, {col:2d}) -> {best_action_name}\n")

    def save_visit_counts_with_coordinates(self, filepath: str, env: MazeEnv) -> None:
        """Save visit counts with coordinates and action descriptions."""
        # Ensure the directory exists.
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            # Write header information.
            f.write("# Visit Counts with State Coordinates and Action Descriptions\n")
            f.write("# ===========================================================\n")
            f.write(f"# Maze Size: {env.size}x{env.size}\n")
            f.write(f"# Number of States: {self.S}\n")
            f.write(f"# Number of Actions: {self.A}\n")
            f.write("# ===========================================================\n\n")

            # Write action mapping.
            f.write("# Action Mapping:\n")
            for i, (dr, dc) in enumerate(env.ACTIONS):
                action_name = env.ACTION_NAMES[i] if i < len(env.ACTION_NAMES) else f"ACTION_{i}"
                f.write(f"#   Action {i}: {action_name} (move: {dr}, {dc})\n")
            f.write("\n")

            # Write visit counts for each state.
            f.write("# State Information and Visit Counts:\n")
            f.write("# Format: StateID (row, col) | TotalVisits | UP | RIGHT | DOWN | LEFT | STAY\n")
            f.write("# -------------------------------------------------------------------------\n")

            for state in range(self.S):
                coord = env.state_to_coord(state)
                row, col = coord

                # Compute total visits.
                total_visits = np.sum(self.N[state, :])

                # Write state info.
                f.write(f"State {state:3d} ({row:2d}, {col:2d}): ")
                f.write(f"Total={total_visits:6d} | ")

                # Write per-action visit counts.
                for action in range(self.A):
                    visit_count = self.N[state, action]
                    f.write(f"{visit_count:6d} ")
                f.write("\n")

            # Write visit count statistics.
            f.write("\n# Visit Counts Statistics:\n")
            f.write("# -----------------------\n")
            total_all_visits = np.sum(self.N)
            f.write(f"Total visits across all states and actions: {total_all_visits}\n")

            # Find the most visited state-action pair.
            if total_all_visits > 0:
                max_visit_state = np.unravel_index(np.argmax(self.N), self.N.shape)[0]
                max_visit_action = np.unravel_index(np.argmax(self.N), self.N.shape)[1]
                max_visit_coord = env.state_to_coord(max_visit_state)
                max_visit_count = np.max(self.N)
                max_action_name = env.ACTION_NAMES[max_visit_action] if max_visit_action < len(
                    env.ACTION_NAMES) else f"ACTION_{max_visit_action}"

                f.write(
                    f"Most visited state-action pair: State {max_visit_state} ({max_visit_coord[0]}, {max_visit_coord[1]}) ")
                f.write(f"Action {max_visit_action} ({max_action_name}) with {max_visit_count} visits\n")
            else:
                f.write("No visits recorded yet.\n")

    def save_q_table_summary(self, filepath: str, env: MazeEnv) -> None:
        """Save summary statistics for the Q-table."""
        # Ensure the directory exists.
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            f.write("# Q-Table and V-Table Summary Statistics\n")
            f.write("# ======================================\n")
            f.write(f"# Maze Size: {env.size}x{env.size}\n")
            f.write(f"# Q-table Shape: {self.Q.shape}\n")
            f.write(f"# V-table Shape: {self.V.shape}\n")
            f.write(f"# Initial Q Value: {self.q_init_value}\n")
            f.write("# ======================================\n\n")

            # Initial Q-value statistics.
            f.write("## Initial Q-values Statistics:\n")
            f.write(f"  Initial Min: {self.initial_q_min}\n")
            f.write(f"  Initial Max: {self.initial_q_max}\n")
            f.write(f"  Initial Mean: {self.initial_q_mean}\n")
            f.write(f"  Initial Std: {self.initial_q_std}\n\n")

            # Initial V-value statistics.
            f.write("## Initial V-values Statistics:\n")
            f.write(f"  Initial Min: {self.initial_v_min}\n")
            f.write(f"  Initial Max: {self.initial_v_max}\n")
            f.write(f"  Initial Mean: {self.initial_v_mean}\n")
            f.write(f"  Initial Std: {self.initial_v_std}\n\n")

            # Final Q-value statistics.
            f.write("## Final Q-values Statistics:\n")
            f.write(f"  Min Q-value: {np.min(self.Q):.6f}\n")
            f.write(f"  Max Q-value: {np.max(self.Q):.6f}\n")
            f.write(f"  Mean Q-value: {np.mean(self.Q):.6f}\n")
            f.write(f"  Std Q-value: {np.std(self.Q):.6f}\n")

            # Final V-value statistics.
            f.write("\n## Final V-values Statistics:\n")
            f.write(f"  Min V-value: {np.min(self.V):.6f}\n")
            f.write(f"  Max V-value: {np.max(self.V):.6f}\n")
            f.write(f"  Mean V-value: {np.mean(self.V):.6f}\n")
            f.write(f"  Std V-value: {np.std(self.V):.6f}\n")

            # Summarize Q-value sign distribution.
            positive_q = np.sum(self.Q > 0)
            negative_q = np.sum(self.Q < 0)
            zero_q = np.sum(self.Q == 0)
            total_q = self.Q.size

            f.write(f"\n## Q-value Distribution:\n")
            f.write(f"  Positive Q-values: {positive_q} ({positive_q / total_q * 100:.2f}%)\n")
            f.write(f"  Negative Q-values: {negative_q} ({negative_q / total_q * 100:.2f}%)\n")
            f.write(f"  Zero Q-values: {zero_q} ({zero_q / total_q * 100:.2f}%)\n")

            # Best policy analysis.
            f.write(f"\n## Best Policy Analysis:\n")
            best_actions = np.argmax(self.Q, axis=1)
            action_counts = {i: np.sum(best_actions == i) for i in range(self.A)}

            for action_idx, count in action_counts.items():
                action_name = env.ACTION_NAMES[action_idx] if action_idx < len(
                    env.ACTION_NAMES) else f"ACTION_{action_idx}"
                percentage = count / self.S * 100
                f.write(f"  {action_name}: {count} states ({percentage:.2f}%)\n")



class QLearningEpsilonGreedy:
    """Independent epsilon-greedy Q-learning baseline."""

    def __init__(self, env: MazeEnv, horizon: int, episodes: int,
                 alpha: float = 0.2, gamma: float = 0.99,
                 epsilon: float = 0.1, seed: Optional[int] = None) -> None:
        self.env = env
        self.H = horizon
        self.K = episodes
        self.S = env.n_states
        self.A = env.n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)

        self.Q = np.zeros((self.S, self.A), dtype=float)
        self.N = np.zeros((self.S, self.A), dtype=int)

    def _select_action(self, state: int, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return self.rng.randrange(self.A)
        q_values = self.Q[state]
        max_q = np.max(q_values)
        best_actions = np.flatnonzero(q_values == max_q)
        return int(self.rng.choice(best_actions))

    def train(self, log_interval: int = 50, record_paths: bool = False) -> TrainingStats:
        rewards: List[float] = []
        successes: List[bool] = []
        episodes_axis: List[int] = []
        episode_paths: List[List[Coordinate]] | None = [] if record_paths else None

        eps = self.epsilon
        for episode in range(1, self.K + 1):
            state_coord = self.env.reset()
            state = self.env.coord_to_state(state_coord)
            if episode_paths is not None:
                episode_coords: List[Coordinate] = [state_coord]

            episode_total_reward = 0.0
            episode_success = False

            for _ in range(self.H):
                action = self._select_action(state, eps)
                next_coord, reward, done, info = self.env.step(action)
                next_state = self.env.coord_to_state(next_coord)

                td_target = reward + self.gamma * np.max(self.Q[next_state])
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += self.alpha * td_error
                self.N[state, action] += 1

                episode_total_reward += reward
                if next_coord == self.env.goal:
                    episode_success = True

                state = next_state
                if episode_paths is not None:
                    episode_coords.append(next_coord)
                if done:
                    break

            if episode_paths is not None:
                episode_paths.append(episode_coords)

            rewards.append(episode_total_reward)
            successes.append(episode_success)
            episodes_axis.append(episode)

            if log_interval and episode % log_interval == 0:
                success_rate = sum(successes[-log_interval:]) / log_interval
                print(f"[EpsilonGreedy] Episode {episode:4d} | eps={eps:.4f} | "
                      f"reward={episode_total_reward:.2f} | success={success_rate:.2f}")

        success_rate = sum(successes) / max(1, len(successes))
        return TrainingStats(
            episodes=episodes_axis,
            rewards=rewards,
            successes=successes,
            success_rate=success_rate,
            episode_paths=episode_paths,
        )

    def evaluate(self, env: MazeEnv, episodes: int = 100) -> Tuple[float, float]:
        success_count = 0
        rewards: List[float] = []

        for _ in range(episodes):
            state_coord = env.reset()
            state = env.coord_to_state(state_coord)
            episode_reward = 0.0
            episode_success = False

            for _ in range(self.H):
                action = int(np.argmax(self.Q[state]))
                next_coord, reward, done, info = env.step(action)
                next_state = env.coord_to_state(next_coord)
                episode_reward += reward
                if next_coord == env.goal:
                    episode_success = True
                state = next_state
                if done:
                    break

            if episode_success:
                success_count += 1
            rewards.append(episode_reward)

        success_rate = success_count / max(1, episodes)
        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        return success_rate, avg_reward

    def rollout(self, env: MazeEnv) -> Tuple[List[Coordinate], float, bool]:
        state_coord = env.reset()
        state = env.coord_to_state(state_coord)
        path: List[Coordinate] = [state_coord]
        total_reward = 0.0
        success = False

        for _ in range(self.H):
            action = int(np.argmax(self.Q[state]))
            next_coord, reward, done, info = env.step(action)
            next_state = env.coord_to_state(next_coord)
            total_reward += reward
            path.append(next_coord)
            if next_coord == env.goal:
                success = True
            state = next_state
            if done:
                break

        return path, total_reward, success

    def greedy_path_from_q(self, env: MazeEnv, deterministic: bool = True) -> List[Coordinate]:
        """Follows the greedy policy derived from the Q-table to get a path."""
        coord = env.reset()
        coords = [coord]
        state = env.coord_to_state(coord)

        for _ in range(self.H):
            q_values = self.Q[state]
            max_value = np.max(q_values)
            max_actions = np.flatnonzero(np.isclose(q_values, max_value))
            if deterministic or len(max_actions) == 1:
                action = int(max_actions[0])
            else:
                action = int(self.rng.choice(max_actions))

            coord, _, done, _ = env.step(action)
            coords.append(coord)
            state = env.coord_to_state(coord)
            if done:
                break

        return coords


def ensure_output_dir() -> str:
    """Ensure the output directory exists and return its path."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = timestamp

    # Create output directory.
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories.
    subdirs = ["plots", "visualizations", "animations", "data", "tables"]
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    return output_dir


def main(**kwargs) -> None:
    """Main entry that manages experiment configuration."""

    # Default configuration.
    config = {
        "experiment_name": "maze_experiment",
        "base_seed": None,
        "output_dir": None,
        "maze_size": 15,
        "maze_seed": 1,
        "env_seed": 42,
        "episodes": 200,
        "horizon": 2000,
        "failure_prob": 0.1,
        "bonus_constant": 1.0,
        "sparse_fraction": 0.01,
        "wall_penalty": -100.0,
        "stay_penalty": -0.2,
        "move_penalty": -0.2,
        "log_interval": 100,
        "eval_episodes": 200,
        "record_paths": True,
    }

    # Use config values, allow overrides via CLI.
    seed = kwargs.get('seed', config.get('base_seed'))
    output_dir = kwargs.get('output_dir', config.get('output_dir'))
    wall_penalty = kwargs.get('wall_penalty', config.get('wall_penalty'))
    stay_penalty = kwargs.get('stay_penalty', config.get('stay_penalty'))
    move_penalty = kwargs.get('move_penalty', config.get('move_penalty'))
    episodes = kwargs.get('episodes', config.get('episodes'))
    horizon = kwargs.get('horizon', config.get('horizon'))

    # Read remaining parameters from config.
    failure_prob = config.get('failure_prob', 0.1)
    bonus_constant = config.get('bonus_constant', 0.1)
    sparse_fraction = config.get('sparse_fraction', 0.01)
    log_interval = config.get('log_interval', 100)
    eval_episodes = config.get('eval_episodes', 200)
    record_paths = config.get('record_paths', True)

    if seed is None:
        seed = random.randint(0, 1000000)

    # Run four experiments with fixed maze seeds and algorithms, then exit.
    if output_dir is None:
        output_dir = ensure_output_dir()
    else:
        os.makedirs(output_dir, exist_ok=True)

    base_sparse_fraction = sparse_fraction
    sparse_one = 1.0
    experiments = [
        {"maze_seed": 1, "algo": "ucb", "variant": "base", "output_name": "ucb",
         "label": "UCB", "sparse_fraction": base_sparse_fraction},
        {"maze_seed": 1, "algo": "ucb", "variant": "sparse1", "output_name": "ucb_h",
         "label": "UCB-H", "sparse_fraction": sparse_one},
        {"maze_seed": 12, "algo": "ucb", "variant": "base", "output_name": "ucb",
         "label": "UCB", "sparse_fraction": base_sparse_fraction},
        {"maze_seed": 12, "algo": "ucb", "variant": "sparse1", "output_name": "ucb_h",
         "label": "UCB-H", "sparse_fraction": sparse_one},
        {"maze_seed": 1, "algo": "eps", "variant": "eps", "output_name": "eps",
         "label": "ε-greedy", "sparse_fraction": base_sparse_fraction},
        {"maze_seed": 12, "algo": "eps", "variant": "eps", "output_name": "eps",
         "label": "ε-greedy", "sparse_fraction": base_sparse_fraction},
    ]

    group_results: Dict[int, Dict[str, TrainingStats]] = {}
    group_results_all: Dict[int, Dict[str, TrainingStats]] = {}
    group_paths: Dict[int, Dict[str, List[Coordinate]]] = {}
    group_paths_all: Dict[int, Dict[str, List[Coordinate]]] = {}

    for exp in experiments:
        group_dir = os.path.join(output_dir, f"maze_seed_{exp['maze_seed']}")
        exp_output_dir = group_dir
        os.makedirs(exp_output_dir, exist_ok=True)
        subdirs = ["plots", "visualizations", "animations", "data", "tables"]
        for subdir in subdirs:
            os.makedirs(os.path.join(exp_output_dir, subdir), exist_ok=True)

        print("=" * 60)
        print(f"Experiment: {exp['label']}")
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Base seed: {seed}")
        print(f"Maze seed: {exp['maze_seed']}")
        print(f"Algorithm: {exp['algo']}")
        print(f"Sparse fraction: {exp['sparse_fraction']}")
        print(f"Wall penalty: {wall_penalty}")
        print(f"Stay penalty: {stay_penalty}")
        print(f"Move penalty: {move_penalty}")
        print(f"Episodes: {episodes}")
        print(f"Horizon: {horizon}")
        print(f"Output dir: {exp_output_dir}")
        print("=" * 60)

        # Use DFS corridor maze for experiments; MazeEnv._generate_maze is skipped here.
        grid = build_corridor_maze(size=config.get('maze_size', 15), seed=exp["maze_seed"])

        env = MazeEnv(
            size=config.get('maze_size', 15),
            horizon=horizon,
            seed=config.get('env_seed', 42),
            grid=grid,
            start=(0, 0),
            goal=(config.get('maze_size', 15) - 1, config.get('maze_size', 15) - 1),
            wall_penalty=wall_penalty,
            stay_penalty=stay_penalty,
            move_penalty=move_penalty
        )

        layout_path = os.path.join(exp_output_dir, "visualizations", "maze_layout.png")
        save_maze_visualization(grid, env.start, env.goal, out_path=layout_path)

        if exp["algo"] == "ucb":
            agent = QLearningUCBHoeffdingSparse(
                env=env,
                horizon=horizon,
                episodes=episodes,
                failure_prob=failure_prob,
                bonus_constant=bonus_constant,
                sparse_fraction=exp["sparse_fraction"],
                seed=seed
            )
        else:
            agent = QLearningEpsilonGreedy(
                env=env,
                horizon=horizon,
                episodes=episodes,
                seed=seed
            )

        stats = agent.train(log_interval=log_interval, record_paths=record_paths)
        overall_success_rate = sum(stats.successes) / max(1, len(stats.successes))
        if exp["algo"] == "ucb":
            group_results.setdefault(exp["maze_seed"], {})[exp["variant"]] = stats
        group_results_all.setdefault(exp["maze_seed"], {})[exp["output_name"]] = stats

        reward_plot_path = os.path.join(
            exp_output_dir, "plots", f"reward_vs_step_{exp['output_name']}.png"
        )
        plot_reward_trace(stats.episodes, stats.rewards, out_path=reward_plot_path)

        success_rate, avg_reward = agent.evaluate(env, episodes=eval_episodes)

        demo_path, demo_reward, success = agent.rollout(env)
        stay_count = sum(1 for i in range(len(demo_path) - 1) if demo_path[i] == demo_path[i + 1])
        print(f"Demo path length: {len(demo_path)} | reward={demo_reward:.4f} | success={success}")

        q_path = agent.greedy_path_from_q(env, deterministic=True)
        q_path_vis = os.path.join(
            exp_output_dir, "visualizations", f"q_greedy_path_{exp['output_name']}.png"
        )
        if exp["algo"] == "eps":
            marker_style = "hollow_blue"
        elif exp["variant"] == "sparse1":
            marker_style = "black_dashed"
        else:
            marker_style = "line"
        save_maze_visualization(
            env.grid, env.start, env.goal, q_path, q_path_vis,
            marker_style=marker_style
        )
        if exp["algo"] == "ucb":
            group_paths.setdefault(exp["maze_seed"], {})[exp["variant"]] = q_path
        group_paths_all.setdefault(exp["maze_seed"], {})[exp["output_name"]] = q_path

        if record_paths and stats.episode_paths:
            anim_path = os.path.join(
                exp_output_dir, "animations", f"episode_paths_{exp['output_name']}.gif"
            )
            save_episode_animation(
                grid, env.start, env.goal,
                episode_paths=stats.episode_paths,
                rewards=stats.rewards,
                out_path=anim_path,
                interval_ms=120
            )

        if exp["algo"] == "ucb":
            q_table_path = os.path.join(
                exp_output_dir, "tables", f"q_table_detailed_{exp['output_name']}.txt"
            )
            agent.save_q_table_with_coordinates(q_table_path, env)

            visit_counts_path = os.path.join(
                exp_output_dir, "tables", f"visit_counts_{exp['output_name']}.txt"
            )
            agent.save_visit_counts_with_coordinates(visit_counts_path, env)

            q_summary_path = os.path.join(
                exp_output_dir, "tables", f"q_table_summary_{exp['output_name']}.txt"
            )
            agent.save_q_table_summary(q_summary_path, env)

        config_path = os.path.join(
            exp_output_dir, "data", f"experiment_config_{exp['output_name']}.json"
        )
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        saved_config = {
            "experiment_info": {
                "name": config.get('experiment_name', 'maze_experiment'),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "base_seed": seed,
                "maze_seed": exp["maze_seed"],
                "env_seed": config.get('env_seed', 42),
                "algorithm": exp["algo"],
                "variant": exp["variant"],
            },
            "environment_params": {
                "maze_size": env.size,
                "horizon": horizon,
                "wall_penalty": wall_penalty,
                "stay_penalty": stay_penalty,
                "move_penalty": move_penalty,
            },
            "algorithm_params": {
                "episodes": episodes,
                "failure_prob": failure_prob,
                "bonus_constant": bonus_constant,
                "sparse_fraction": exp["sparse_fraction"],
            },
            "training_params": {
                "log_interval": log_interval,
                "eval_episodes": eval_episodes,
                "record_paths": record_paths,
            },
            "results": {
                "final_success_rate": stats.success_rate,
                "overall_success_rate": overall_success_rate,
                "evaluation_success_rate": success_rate,
                "evaluation_avg_reward": avg_reward,
                "demo_success": success,
                "demo_stay_count": stay_count,
            }
        }

        import json
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(saved_config, f, indent=2, ensure_ascii=False)

    for maze_seed, results in group_results_all.items():
        if "ucb" not in results or "ucb_h" not in results or "eps" not in results:
            continue
        group_dir = os.path.join(output_dir, f"maze_seed_{maze_seed}")
        compare_plot_path = os.path.join(group_dir, "plots", "reward_vs_step_compare_three.png")
        plot_reward_comparison_three(
            results["ucb"], "Proposed",
            results["ucb_h"], "UCB-H",
            results["eps"], "ε-greedy",
            out_path=compare_plot_path
        )

    for maze_seed, paths in group_paths_all.items():
        if "ucb" not in paths or "ucb_h" not in paths or "eps" not in paths:
            continue
        group_dir = os.path.join(output_dir, f"maze_seed_{maze_seed}")
        grid = build_corridor_maze(size=config.get('maze_size', 15), seed=maze_seed)
        start = (0, 0)
        goal = (config.get('maze_size', 15) - 1, config.get('maze_size', 15) - 1)
        compare_path_out = os.path.join(group_dir, "visualizations", "q_greedy_path_compare_three.png")
        save_maze_visualization_compare_three(
            grid, start, goal,
            paths["ucb"], "Proposed",
            paths["ucb_h"], "UCB-H",
            paths["eps"], "ε-greedy",
            out_path=compare_path_out
        )

    return

    # Create output directory.
    if output_dir is None:
        output_dir = ensure_output_dir()
    else:
        os.makedirs(output_dir, exist_ok=True)
        subdirs = ["plots", "visualizations", "animations", "data", "tables"]
        for subdir in subdirs:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    print("=" * 60)
    print("迷宫强化学习实验 - 稀疏奖励感知版本")
    print(f"实验时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"基础随机种子: {seed}")
    print(f"撞墙惩罚: {wall_penalty}")
    print(f"原地不动惩罚: {stay_penalty}")
    print(f"普通移动惩罚: {move_penalty}")
    print(f"训练回合数: {episodes}")
    print(f"Horizon: {horizon}")
    print(f"输出目录: {output_dir}")
    print("动作空间: UP, RIGHT, DOWN, LEFT, STAY (5个动作)")
    print("随机数生成: 预先生成随机数池，顺序抽取")
    s_init = sparse_fraction * horizon
    print(f"Q值初始化: {s_init} (按稀疏步数 s)")
    print("V值初始化: 0.0")
    print("=" * 60)

    # Maze generation: fixed seed (reproducible layout).
    # Alternative path if running the single-experiment block below (currently unreachable).
    # Still uses the same corridor maze generator for consistent layouts.
    grid = build_corridor_maze(size=config.get('maze_size', 15), seed=config.get('maze_seed', 7))

    # Environment init: fixed seed to keep transitions deterministic.
    env = MazeEnv(
        size=config.get('maze_size', 15),
        horizon=horizon,
        seed=config.get('env_seed', 42),
        grid=grid,
        start=(0, 0),
        goal=(config.get('maze_size', 15) - 1, config.get('maze_size', 15) - 1),
        wall_penalty=wall_penalty,
        stay_penalty=stay_penalty,
        move_penalty=move_penalty
    )

    print(f"初始 {env.size}x{env.size} 迷宫布局 (S=起点, G=终点):")
    print(env.render())
    print(f"起点: {env.start}, 终点: {env.goal}")

    # Save maze layout.
    layout_path = os.path.join(output_dir, "visualizations", "maze_layout.png")
    save_maze_visualization(grid, env.start, env.goal, out_path=layout_path)
    print(f"迷宫可视化保存至: {layout_path}")

    # Algorithm init: use the specified seed.
    agent = QLearningUCBHoeffdingSparse(
        env=env,
        horizon=horizon,
        episodes=episodes,
        failure_prob=failure_prob,
        bonus_constant=bonus_constant,
        sparse_fraction=sparse_fraction,
        seed=seed
    )

    print(f"开始训练，总回合数: {episodes}")
    print(f"初始Q值: {agent.q_init_value}")
    print("初始V值: 0.0")
    print("-" * 40)

    stats = agent.train(log_interval=log_interval, record_paths=record_paths)

    # Compute overall success rate.
    overall_success_rate = sum(stats.successes) / max(1, len(stats.successes))

    print("\n训练完成!")
    print(f"最终成功率: {stats.success_rate:.4f}")
    print(f"总体成功率: {overall_success_rate:.4f}")

    # Save reward curve plot.
    reward_plot_path = os.path.join(output_dir, "plots", "reward_vs_step.png")
    plot_reward_trace(stats.episodes, stats.rewards, out_path=reward_plot_path)
    print(f"奖励曲线图保存至: {reward_plot_path}")

    # Evaluate the trained policy.
    print("\n开始评估训练好的策略...")
    success_rate, avg_reward = agent.evaluate(env, episodes=eval_episodes)
    print(f"评估结果 - 成功率: {success_rate:.4f}, 平均奖励: {avg_reward:.4f}")

    # Demonstrate the greedy policy once.
    print("\n演示最优策略路径...")
    demo_path, demo_reward, success = agent.rollout(env)
    stay_count = sum(1 for i in range(len(demo_path) - 1) if demo_path[i] == demo_path[i + 1])
    print(f"演示路径长度: {len(demo_path)}, 总奖励: {demo_reward:.4f}")
    print(f"是否到达目标: {'是' if success else '否'}")
    print(f"原地不动次数: {stay_count}")
    print("路径轨迹:")
    print(env.render(demo_path))

    # Build the greedy path from the Q-table.
    print("\n根据Q表生成最优路径...")
    q_path = agent.greedy_path_from_q(env, deterministic=True)
    print(env.render(q_path))
    q_path_vis = os.path.join(output_dir, "visualizations", "q_greedy_path.png")
    save_maze_visualization(env.grid, env.start, env.goal, q_path, q_path_vis)
    print(f"Q表最优路径可视化保存至: {q_path_vis}")

    # Save animation.
    if record_paths and stats.episode_paths:
        anim_path = os.path.join(output_dir, "animations", "episode_paths.gif")
        save_episode_animation(
            grid, env.start, env.goal,
            episode_paths=stats.episode_paths,
            rewards=stats.rewards,
            out_path=anim_path,
            interval_ms=120
        )
        print(f"训练过程动画保存至: {anim_path}")

    # Save Q-table and visit counts.
    q_table_path = os.path.join(output_dir, "tables", "q_table_detailed.txt")
    agent.save_q_table_with_coordinates(q_table_path, env)
    print(f"详细Q表保存至: {q_table_path}")

    visit_counts_path = os.path.join(output_dir, "tables", "visit_counts.txt")
    agent.save_visit_counts_with_coordinates(visit_counts_path, env)
    print(f"访问次数统计保存至: {visit_counts_path}")

    q_summary_path = os.path.join(output_dir, "tables", "q_table_summary.txt")
    agent.save_q_table_summary(q_summary_path, env)
    print(f"Q表统计摘要保存至: {q_summary_path}")

    # Save experiment configuration.
    config_path = os.path.join(output_dir, "data", "experiment_config.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # Save the full configuration.
    saved_config = {
        "experiment_info": {
            "name": config.get('experiment_name', 'maze_experiment'),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_seed": seed,
            "maze_seed": config.get('maze_seed', 7),
            "env_seed": config.get('env_seed', 42),
        },
        "environment_params": {
            "maze_size": env.size,
            "horizon": horizon,
            "wall_penalty": wall_penalty,
            "stay_penalty": stay_penalty,
            "move_penalty": move_penalty,
        },
        "algorithm_params": {
            "episodes": episodes,
            "failure_prob": failure_prob,
            "bonus_constant": bonus_constant,
            "sparse_fraction": sparse_fraction,
        },
        "training_params": {
            "log_interval": log_interval,
            "eval_episodes": eval_episodes,
            "record_paths": record_paths,
        },
        "results": {
            "final_success_rate": stats.success_rate,
            "overall_success_rate": overall_success_rate,
            "evaluation_success_rate": success_rate,
            "evaluation_avg_reward": avg_reward,
            "demo_success": success,
            "demo_stay_count": stay_count,
        }
    }

    import json
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(saved_config, f, indent=2, ensure_ascii=False)

    print(f"完整实验配置已保存至: {config_path}")


# Adjust the main invocation block.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="迷宫强化学习实验 - 稀疏奖励感知版本")

    # Original arguments (backward compatible).
    parser.add_argument("--seed", type=int, default=None,
                        help="基础随机种子（默认为随机生成）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认为maze_experiment_时间戳）")
    parser.add_argument("--wall_penalty", type=float, default=None,
                        help="撞墙惩罚值")
    parser.add_argument("--stay_penalty", type=float, default=None,
                        help="原地不动惩罚值")
    parser.add_argument("--move_penalty", type=float, default=None,
                        help="普通移动惩罚值")
    parser.add_argument("--episodes", type=int, default=200,
                        help="训练回合数（默认200）")
    parser.add_argument("--horizon", type=int, default=2000,
                        help="每个episode的最大步数（默认2000）")

    args = parser.parse_args()

    # Filter out None values and pass only provided args.
    kwargs = {k: v for k, v in vars(args).items() if v is not None}

    main(**kwargs)
