import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1) Maze (your maze)
# ============================================================
MAZE_STR = """
111111111111111111111111
100000100000100001000001
100000100000100000000001
100000100000100001000001
111110111011101111111111
100010000000000000010001
100010000000000000000001
100010011101111110010001
100000010000000010011111
111110010000000010010001
100010011111111110010001
100000010001000010010001
100010010001000000010001
111110011011111110011011
100000000000000000000001
100000000000000000000001
110111101111111011111111
100001000000100001000001
100001000000100000000001
100001000000100001000001
111111111111111111111111
""".strip().splitlines()

def parse_maze(lines):
    H = len(lines)
    W = len(lines[0])
    maze = np.zeros((H, W), dtype=int)
    for y, row in enumerate(lines):
        if len(row) != W:
            raise ValueError("All rows must have the same width.")
        for x, ch in enumerate(row):
            if ch not in ("0", "1"):
                raise ValueError("Maze must be '0'/'1' only.")
            maze[y, x] = int(ch)
    return maze

maze = parse_maze(MAZE_STR)
H, W = maze.shape

def free_cells(maze):
    ys, xs = np.where(maze == 0)
    return list(zip(xs.tolist(), ys.tolist()))  # (x,y)

FREE = free_cells(maze)
FREE_SET = set(FREE)

# ============================================================
# 2) Fixed goals (Task 1..50)  ※(x,y) = 0-based
# ============================================================
TASK_GOALS = {
    1:  (18, 1),
    2:  (3, 2),
    3:  (3, 18),
    4:  (20, 18),
    5:  (14, 1),
    6:  (2, 2),
    7:  (20, 2),
    8:  (21, 6),
    9:  (20, 11),
    10: (1, 11),
    11: (8, 18),
    12: (15, 18),
    13: (22, 17),
    14: (8, 2),
    15: (1, 6),
    16: (14, 8),
    17: (9, 11),
    18: (10, 12),
    19: (12, 12),
    20: (13, 11),
    21: (20, 3),
    22: (22, 11),
    23: (14, 9),
    24: (9, 8),
    25: (12, 9),
    26: (2, 19),
    27: (11, 2),
    28: (20, 7),
    29: (20, 10),
    30: (3, 10),
    31: (3, 7),
    32: (15, 3),
    33: (15, 19),
    34: (10, 19),
    35: (5, 1),
    36: (19, 17),
    37: (13, 17),
    38: (10, 17),
    39: (3, 17),
    40: (3, 12),
    41: (3, 5),
    42: (9, 1),
    43: (15, 1),
    44: (6, 19),
    45: (19, 19),
    46: (2, 3),
    47: (21, 1),
    48: (21, 5),
    49: (14, 12),
    50: (2, 8),
}

# Task 1..50 を順番通りのリストにする
tasks_goals_fixed = [TASK_GOALS[i] for i in range(1, 51)]

# sanity check（壁の上にゴール置いてないか）
for i, g in enumerate(tasks_goals_fixed, start=1):
    if g not in FREE_SET:
        raise ValueError(f"Task {i} goal {g} is not a free cell!")

# ============================================================
# 3) Environment (random start, goal is 1 cell, reward: success=1 else 0)
# ============================================================
class MazeEnv:
    """
    Discrete grid environment:
    - Start: uniform random over free cells excluding goal
    - Goal: single cell (x,y)
    - Reward: 1 on reaching goal else 0
    - Episode ends: reach goal OR max_steps Hmax
    - Walls block motion (stay)
    """
    def __init__(self, maze, goal, gamma=0.95, seed=0):
        self.maze = maze
        self.H, self.W = maze.shape
        self.goal = tuple(goal)
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

        self.moves = {
            0: (0, -1),  # up
            1: (1, 0),   # right
            2: (0, 1),   # down
            3: (-1, 0),  # left
        }

        if self.goal not in FREE_SET:
            raise ValueError(f"Goal {self.goal} is not a free cell.")

        self.start_candidates = [(x, y) for (x, y) in FREE if (x, y) != self.goal]
        if len(self.start_candidates) == 0:
            raise ValueError("No start candidates.")
        self.pos = self.start_candidates[0]

    @property
    def n_states(self):
        return self.H * self.W

    @property
    def n_actions(self):
        return 4

    def state_id(self, pos):
        x, y = pos
        return y * self.W + x

    def in_bounds(self, x, y):
        return 0 <= x < self.W and 0 <= y < self.H

    def is_wall(self, x, y):
        return self.maze[y, x] == 1

    def reset(self):
        self.pos = self.start_candidates[int(self.rng.integers(0, len(self.start_candidates)))]
        return self.state_id(self.pos)

    def step(self, a):
        x, y = self.pos
        dx, dy = self.moves[a]
        nx, ny = x + dx, y + dy

        if (not self.in_bounds(nx, ny)) or self.is_wall(nx, ny):
            nx, ny = x, y

        self.pos = (nx, ny)
        done = (self.pos == self.goal)
        r = 1.0 if done else 0.0
        return self.state_id(self.pos), r, done

# ============================================================
# 4) Action selection helpers
# ============================================================
def greedy_action(Q, s, rng):
    qs = Q[s]
    mx = np.max(qs)
    cand = np.flatnonzero(qs == mx)
    return int(rng.choice(cand))

def epsilon_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(0, Q.shape[1]))
    return greedy_action(Q, s, rng)

def softmax_stable(values, tau):
    v = np.asarray(values, dtype=float)
    z = tau * v
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

# ============================================================
# 5) One episode routines
# ============================================================
def q_learning_episode_greedy(env, Q, alpha, gamma, max_steps, rng):
    s = env.reset()
    G = 0.0
    disc = 1.0

    for _t in range(max_steps):
        a = greedy_action(Q, s, rng)
        ns, r, done = env.step(a)

        Q[s, a] += alpha * ((r + gamma * np.max(Q[ns])) - Q[s, a])

        G += disc * r
        disc *= gamma
        s = ns
        if done:
            break
    return G

def q_learning_episode_eps(env, Q, alpha, gamma, max_steps, eps, rng):
    """Baseline用：ε-greedy Q-learning 1 episode"""
    s = env.reset()
    G = 0.0
    disc = 1.0

    for _t in range(max_steps):
        a = epsilon_greedy(Q, s, eps, rng)
        ns, r, done = env.step(a)

        Q[s, a] += alpha * ((r + gamma * np.max(Q[ns])) - Q[s, a])

        G += disc * r
        disc *= gamma
        s = ns
        if done:
            break
    return G

def q_learning_episode_pi_reuse(env, Q_new, Q_past, alpha, gamma, max_steps,
                               psi=1.0, nu=0.95, rng=None):
    assert rng is not None
    s = env.reset()
    G = 0.0
    disc = 1.0

    for t in range(max_steps):
        psi_h = psi * (nu ** t)
        psi_h = float(np.clip(psi_h, 0.0, 1.0))

        if rng.random() < psi_h:
            a = greedy_action(Q_past, s, rng)
        else:
            eps = 1.0 - psi_h
            a = epsilon_greedy(Q_new, s, eps, rng)

        ns, r, done = env.step(a)
        Q_new[s, a] += alpha * ((r + gamma * np.max(Q_new[ns])) - Q_new[s, a])

        G += disc * r
        disc *= gamma
        s = ns
        if done:
            break

    return G

# ============================================================
# 6) Baseline Q-learning runner (curve)
# ============================================================
def q_learning_baseline_curve(env,
                              K=2000, Hmax=100,
                              alpha=0.05, gamma=0.95,
                              eps_start=1.0, eps_end=0.05,
                              seed=0):
    """
    Baseline: Q-learning with epsilon-greedy
    Returns raw G per episode (length K)
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions), dtype=float)
    curve = np.zeros(K, dtype=float)

    for k in range(K):
        # 線形減衰（好みで変更OK）
        if K <= 1:
            eps = eps_end
        else:
            frac = 1.0 - (k / (K - 1))
            eps = eps_end + (eps_start - eps_end) * frac

        G = q_learning_episode_eps(env, Q, alpha, gamma, Hmax, eps, rng)
        curve[k] = G

    return Q, curve

# ============================================================
# 7) PRQ-Learning + record G per episode (raw)
# ============================================================
def prq_learning(env, library_Q,
                 K=2000, Hmax=100,
                 alpha=0.05, gamma=0.95,
                 psi=1.0, nu=0.95,
                 tau0=0.0, delta_tau=0.05,
                 seed=0,
                 record_curve=False):
    rng = np.random.default_rng(seed)
    n = len(library_Q)

    Q_new = np.zeros((env.n_states, env.n_actions), dtype=float)
    W = np.zeros(n + 1, dtype=float)     # running avg gain per policy choice
    U = np.zeros(n + 1, dtype=int)       # counts

    tau = float(tau0)
    total_G = 0.0

    curve = np.zeros(K, dtype=float) if record_curve else None

    for k in range(K):
        p = softmax_stable(W, tau)
        chosen = int(rng.choice(np.arange(n + 1), p=p))

        if chosen == 0:
            G = q_learning_episode_greedy(env, Q_new, alpha, gamma, Hmax, rng)
        else:
            G = q_learning_episode_pi_reuse(
                env, Q_new, library_Q[chosen - 1],
                alpha, gamma, Hmax, psi=psi, nu=nu, rng=rng
            )

        W[chosen] = (W[chosen] * U[chosen] + G) / (U[chosen] + 1)
        U[chosen] += 1

        tau += delta_tau
        total_G += G
        if record_curve:
            curve[k] = G

    avg_gain_overall = total_G / K
    return Q_new, W, U, avg_gain_overall, curve

# ============================================================
# 8) PLPR
# ============================================================
def plpr_run(tasks_goals,
             delta=0.25,
             K=2000, Hmax=100,
             alpha=0.05, gamma=0.95,
             psi=1.0, nu=0.95,
             tau0=0.0, delta_tau=0.05,
             seed=0,
             run_label="",
             record_task_indices_1based=(13, 30, 50)):
    rng = np.random.default_rng(seed)

    library = []
    gains = []
    lib_sizes = []
    added_flags = []
    core_goal_indices = []

    record_set0 = {i - 1 for i in record_task_indices_1based}
    curves_dict = {}
    goals_of_recorded = {}
    env_seeds_recorded = {}

    for t, goal in enumerate(tasks_goals):
        env_seed = int(rng.integers(0, 10**9))
        env = MazeEnv(maze, goal=goal, gamma=gamma, seed=env_seed)

        record_this = (t in record_set0)

        Q_new, W, U, avg_gain, curve = prq_learning(
            env, library,
            K=K, Hmax=Hmax,
            alpha=alpha, gamma=gamma,
            psi=psi, nu=nu,
            tau0=tau0, delta_tau=delta_tau,
            seed=int(rng.integers(0, 10**9)),
            record_curve=record_this
        )

        if record_this:
            idx1 = t + 1
            curves_dict[idx1] = curve
            goals_of_recorded[idx1] = goal
            env_seeds_recorded[idx1] = env_seed  # baselineと揃えるため保存

        W_omega = float(W[0])
        W_max = float(np.max(W[1:])) if len(library) > 0 else float("-inf")

        add = (len(library) == 0) or (W_max < delta * W_omega)
        if add:
            library.append(Q_new)
            core_goal_indices.append(t)

        gains.append(avg_gain)
        lib_sizes.append(len(library))
        added_flags.append(add)

        print(f"{run_label} Task {t+1:02d}/{len(tasks_goals)} "
              f"|L|={len(library):02d} add={add} "
              f"WΩ={W_omega:.4f} Wmax={W_max:.4f} avgGain={avg_gain:.4f} goal={goal}")

    return (np.array(gains), np.array(lib_sizes), np.array(added_flags),
            core_goal_indices, curves_dict, goals_of_recorded, env_seeds_recorded)

# ============================================================
# 9) Visualization  ★グリッド線入り版
# ============================================================
def plot_goals_on_maze(maze, goals, title="(a) 50 goal points",
                       grid=True, grid_color="0.7", grid_lw=0.6):
    H, W = maze.shape
    fig, ax = plt.subplots(figsize=(7, 6))

    img = np.zeros((H, W, 3), dtype=float)
    img[maze == 0] = (1, 1, 1)
    img[maze == 1] = (0, 0, 0)

    # 線が滲みにくい設定
    ax.imshow(img, interpolation="none")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    # --- マスの間に線（セル境界グリッド）を入れる ---
    if grid:
        ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax.grid(which="minor", color=grid_color, linewidth=grid_lw)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)

    for i, (x, y) in enumerate(goals, start=1):
        ax.text(
            x, y, str(i),
            ha="center", va="center", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.12",
                      facecolor="white", edgecolor="gray", linewidth=0.5)
        )
    plt.show()

def cumulative_mean(x):
    x = np.asarray(x, dtype=float)
    return np.cumsum(x) / np.arange(1, len(x) + 1)

def plot_compare_paper_like(prq_curves, rl_curves, title, tick_every=200):
    """
    prq_curves / rl_curves: list of raw G curves (each length K)
    Plot cumulative mean, with mean±std at discrete ticks.
    """
    Xp = np.stack([cumulative_mean(c) for c in prq_curves], axis=0)
    Xr = np.stack([cumulative_mean(c) for c in rl_curves], axis=0)

    mp, sp = Xp.mean(axis=0), Xp.std(axis=0)
    mr, sr = Xr.mean(axis=0), Xr.std(axis=0)

    K = mp.shape[0]
    xs = np.arange(tick_every, K + 1, tick_every)
    idx = xs - 1

    plt.figure(figsize=(8, 4))
    plt.errorbar(xs, mp[idx], yerr=sp[idx], fmt='o-', capsize=3, label="PRQ (PLPR inside)")
    plt.errorbar(xs, mr[idx], yerr=sr[idx], fmt='s-', capsize=3, label="Baseline Q-learning (ε-greedy)")
    plt.title(title)
    plt.xlabel("Trials")
    plt.ylabel("Cumulative mean gain")
    plt.grid(True)
    plt.legend()
    plt.show()

# ============================================================
# 10) Main
# ============================================================
def main():
    DELTA = 0.25
    NUM_TASKS = 50
    RUNS = 1

    K = 2000
    Hmax = 100
    gamma = 0.95
    alpha = 0.05

    # PRQ parameters
    psi = 1.0
    nu = 0.95
    tau0 = 0.0
    delta_tau = 0.05

    # Baseline Q-learning exploration
    eps_start = 1.0
    eps_end = 0.05

    # 学習曲線を出すタスク（この3つだけBaselineも回す）
    RECORD_TASKS = (13, 30, 50)

    plot_goals_on_maze(
        maze, tasks_goals_fixed,
        title="(a) 50 fixed goal points (Task 1..50)",
        grid=True, grid_color="0.7", grid_lw=0.6
    )

    print("\n==============================")
    print(f"Start δ={DELTA} (fixed), RUNS={RUNS}, record tasks={RECORD_TASKS}")
    print("==============================")

    # PRQ(PLPR)の学習曲線用
    prq_curves_all = {t: [] for t in RECORD_TASKS}
    # Baseline Q-learningの学習曲線用
    rl_curves_all = {t: [] for t in RECORD_TASKS}

    for r in range(RUNS):
        # ----- PRQ(PLPR): 50タスク全部回してライブラリを作りつつ、指定タスクだけcurve保存 -----
        gains, lib_sizes, added_flags, core_idx, prq_curves_dict, goals_dict, env_seeds_dict = plpr_run(
            tasks_goals_fixed,
            delta=DELTA,
            K=K, Hmax=Hmax,
            alpha=alpha, gamma=gamma,
            psi=psi, nu=nu,
            tau0=tau0, delta_tau=delta_tau,
            seed=2000 + r,
            run_label=f"[run {r+1:02d}/{RUNS:02d}]",
            record_task_indices_1based=RECORD_TASKS
        )

        # PRQ curve格納
        for t in RECORD_TASKS:
            if t in prq_curves_dict and prq_curves_dict[t] is not None:
                prq_curves_all[t].append(prq_curves_dict[t])

        # ----- Baseline: RECORD_TASKS の3タスクだけ回す -----
        for t in RECORD_TASKS:
            goal = tasks_goals_fixed[t - 1]
            env_seed = env_seeds_dict[t]  # PRQと同じ開始乱数系列に合わせる
            env = MazeEnv(maze, goal=goal, gamma=gamma, seed=env_seed)

            _, curve_rl = q_learning_baseline_curve(
                env,
                K=K, Hmax=Hmax,
                alpha=alpha, gamma=gamma,
                eps_start=eps_start, eps_end=eps_end,
                seed=9999 + 100 * r + t  # Q学習側の行動選択乱数
            )
            rl_curves_all[t].append(curve_rl)

    # ----- Compare plot (Taskごと) -----
    for t in RECORD_TASKS:
        if len(prq_curves_all[t]) == 0 or len(rl_curves_all[t]) == 0:
            print(f"[WARN] no curves collected for Task {t}")
            continue
        goal = tasks_goals_fixed[t - 1]
        plot_compare_paper_like(
            prq_curves_all[t],
            rl_curves_all[t],
            title=f"PRQ vs Baseline - Task {t}/{NUM_TASKS} goal={goal} (δ={DELTA}, runs={RUNS})",
            tick_every=200
        )

if __name__ == "__main__":
    main()
