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
# 2) Fixed goals (Task 1..15)  ※(x,y) = 0-based
# ============================================================
TASK_GOALS = {
    1:  (2, 2),
    2:  (1, 11),
    3:  (22, 17),
    4:  (9, 11),
    5:  (12, 12),
    6:  (14, 9),
    7:  (2, 19),
    8:  (20, 10),
    9:  (15, 19),
    10: (10, 19),
    11: (3, 5),
    12: (9, 1),
    13: (15, 1),
    14: (21, 1),
    15: (21, 5),
}
tasks_goals_fixed = [TASK_GOALS[i] for i in range(1, 16)]

for i, g in enumerate(tasks_goals_fixed, start=1):
    if g not in FREE_SET:
        raise ValueError(f"Task {i} goal {g} is not a free cell!")

# ============================================================
# 3) Environment  ★報酬設計：goal_reward + step_cost (+ wall_penalty)
# ============================================================
class MazeEnv:
    """
    Reward design:
      - Each step: step_cost (negative)
      - If action hits wall/out-of-bounds (no movement): add wall_penalty (negative)
      - If reach goal: reward = goal_reward (override step penalties for that step)
    """
    def __init__(self, maze, goal, gamma=0.95, seed=0,
                 goal_reward=1.0, step_cost=-0.01, wall_penalty=-0.05):
        self.maze = maze
        self.H, self.W = maze.shape
        self.goal = tuple(goal)
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

        self.goal_reward = float(goal_reward)
        self.step_cost = float(step_cost)
        self.wall_penalty = float(wall_penalty)

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

        blocked = False
        if (not self.in_bounds(nx, ny)) or self.is_wall(nx, ny):
            nx, ny = x, y
            blocked = True

        self.pos = (nx, ny)
        done = (self.pos == self.goal)

        if done:
            r = self.goal_reward
        else:
            r = self.step_cost + (self.wall_penalty if blocked else 0.0)

        return self.state_id(self.pos), float(r), done

# ============================================================
# 4) Action selection helpers
# ============================================================
def greedy_action(Q, s, rng):
    qs = Q[s]
    mx = np.max(qs)
    cand = np.flatnonzero(qs == mx)
    return int(rng.choice(cand))

def epsilon_greedy_action(Q, s, eps, rng):
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
# 4.5) Bandit selection (ONLY Boltzmann here)
# ============================================================
def bandit_select_arm_boltzmann(W, rng, tau=1.0):
    W = np.asarray(W, dtype=float)
    p = softmax_stable(W, tau)
    return int(rng.choice(np.arange(W.shape[0]), p=p))

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
    s = env.reset()
    G = 0.0
    disc = 1.0
    for _t in range(max_steps):
        a = epsilon_greedy_action(Q, s, eps, rng)
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
        psi_h = float(np.clip(psi * (nu ** t), 0.0, 1.0))

        if rng.random() < psi_h:
            a = greedy_action(Q_past, s, rng)
        else:
            eps = 1.0 - psi_h
            a = epsilon_greedy_action(Q_new, s, eps, rng)

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
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions), dtype=float)
    curve = np.zeros(K, dtype=float)

    for k in range(K):
        if K <= 1:
            eps = eps_end
        else:
            frac = 1.0 - (k / (K - 1))
            eps = eps_end + (eps_start - eps_end) * frac

        curve[k] = q_learning_episode_eps(env, Q, alpha, gamma, Hmax, eps, rng)

    return Q, curve

# ============================================================
# 7) PRQ-Learning (bandit = Boltzmann ONLY)
# ============================================================
def prq_learning_boltzmann(env, library_Q,
                           K=2000, Hmax=100,
                           alpha=0.05, gamma=0.95,
                           psi=1.0, nu=0.95,
                           tau0=0.05, delta_tau=0.05,
                           seed=0,
                           record_curve=False):
    rng = np.random.default_rng(seed)
    n = len(library_Q)

    Q_new = np.zeros((env.n_states, env.n_actions), dtype=float)
    W = np.zeros(n + 1, dtype=float)
    U = np.zeros(n + 1, dtype=int)

    tau = float(tau0)
    total_G = 0.0
    curve = np.zeros(K, dtype=float) if record_curve else None

    for k in range(K):
        chosen = bandit_select_arm_boltzmann(W, rng, tau=tau)

        if chosen == 0:
            G = q_learning_episode_greedy(env, Q_new, alpha, gamma, Hmax, rng)
        else:
            G = q_learning_episode_pi_reuse(
                env, Q_new, library_Q[chosen - 1],
                alpha, gamma, Hmax, psi=psi, nu=nu, rng=rng
            )

        W[chosen] = (W[chosen] * U[chosen] + G) / (U[chosen] + 1)
        U[chosen] += 1

        tau += float(delta_tau)
        total_G += G
        if record_curve:
            curve[k] = G

    avg_gain_overall = total_G / K
    return Q_new, W, U, avg_gain_overall, curve

# ============================================================
# 8) PLPR (Boltzmann bandit)
# ============================================================
def plpr_run_boltzmann(tasks_goals,
                       delta=0.25,
                       K=2000, Hmax=100,
                       alpha=0.05, gamma=0.95,
                       psi=1.0, nu=0.95,
                       tau0=0.05,                # ★固定
                       delta_tau=0.05,
                       env_seeds=None,
                       prq_seeds=None,
                       run_label="",
                       record_task_indices_1based=(3, 5, 11)):
    num_tasks = len(tasks_goals)
    if env_seeds is None or prq_seeds is None:
        raise ValueError("Please provide env_seeds and prq_seeds for fair grid search.")
    if len(env_seeds) != num_tasks or len(prq_seeds) != num_tasks:
        raise ValueError("env_seeds/prq_seeds must have length = num_tasks.")

    library = []
    gains = []
    lib_sizes = []
    added_flags = []
    core_goal_indices = []

    record_set0 = {i - 1 for i in record_task_indices_1based}
    curves_dict = {}
    goals_of_recorded = {}

    for t, goal in enumerate(tasks_goals):
        env = MazeEnv(maze, goal=goal, gamma=gamma, seed=int(env_seeds[t]))
        record_this = (t in record_set0)

        Q_new, W, U, avg_gain, curve = prq_learning_boltzmann(
            env, library,
            K=K, Hmax=Hmax,
            alpha=alpha, gamma=gamma,
            psi=psi, nu=nu,
            tau0=float(tau0), delta_tau=float(delta_tau),
            seed=int(prq_seeds[t]),
            record_curve=record_this
        )

        if record_this:
            idx1 = t + 1
            curves_dict[idx1] = curve
            goals_of_recorded[idx1] = goal

        W_omega = float(W[0])
        W_max = float(np.max(W[1:])) if len(library) > 0 else float("-inf")

        add = (len(library) == 0) or (W_max < delta * W_omega)
        if add:
            library.append(Q_new)
            core_goal_indices.append(t)

        gains.append(avg_gain)
        lib_sizes.append(len(library))
        added_flags.append(add)

        print(f"{run_label} (tau0={tau0:.3f}, dTau={delta_tau:.3f}) Task {t+1:02d}/{num_tasks} "
              f"|L|={len(library):02d} add={add} "
              f"WΩ={W_omega:.4f} Wmax={W_max:.4f} avgGain={avg_gain:.4f} goal={goal}")

    return (np.array(gains), np.array(lib_sizes), np.array(added_flags),
            core_goal_indices, curves_dict, goals_of_recorded)

# ============================================================
# 9) Visualization
# ============================================================
def plot_goals_on_maze(maze, goals, title="(a) goal points",
                       grid=True, grid_color="0.7", grid_lw=0.6):
    H, W = maze.shape
    fig, ax = plt.subplots(figsize=(7, 6))

    img = np.zeros((H, W, 3), dtype=float)
    img[maze == 0] = (1, 1, 1)
    img[maze == 1] = (0, 0, 0)

    ax.imshow(img, interpolation="none")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

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

def plot_compare_multi_paper_like(prq_curves_by_label, rl_curves, title, tick_every=200):
    plt.figure(figsize=(10, 4))

    fmts = ["o-", "^-", "d-", "v-", "x-", "+-", "*-", "s-", "p-"]
    for i, (label, curves) in enumerate(prq_curves_by_label.items()):
        X = np.stack([cumulative_mean(c) for c in curves], axis=0)
        m, s = X.mean(axis=0), X.std(axis=0)
        K = m.shape[0]
        xs = np.arange(tick_every, K + 1, tick_every)
        idx = xs - 1
        fmt = fmts[i % len(fmts)]
        plt.errorbar(xs, m[idx], yerr=s[idx], fmt=fmt, capsize=3, label=label)

    Xr = np.stack([cumulative_mean(c) for c in rl_curves], axis=0)
    mr, sr = Xr.mean(axis=0), Xr.std(axis=0)
    K = mr.shape[0]
    xs = np.arange(tick_every, K + 1, tick_every)
    idx = xs - 1
    plt.errorbar(xs, mr[idx], yerr=sr[idx], fmt="h-", capsize=3,
                 label="Baseline Q-learning (ε-greedy)")

    plt.title(title)
    plt.xlabel("Trials")
    plt.ylabel("Cumulative mean gain")
    plt.grid(True)
    plt.legend()
    plt.show()

# ============================================================
# 10) Main: Boltzmann bandit ONLY + grid search on delta_tau ONLY
#     ★ tau0 は 0.05 固定、delta_tau だけ探索
# ============================================================
def main():
    DELTA = 0.25
    RUNS = 5

    K = 2000
    Hmax = 100
    gamma = 0.95
    alpha = 0.05

    # PRQ parameters
    psi = 1.0
    nu = 0.95

    # ★ tau0 固定
    TAU0_FIXED = 0.05
    # ★ delta_tau だけグリッド
    DELTA_TAU_LIST = [0.0, 0.005, 0.01, 0.02, 0.05]

    # Baseline Q-learning exploration
    eps_start = 1.0
    eps_end = 0.05

    NUM_TASKS = len(tasks_goals_fixed)   # 15
    RECORD_TASKS = (3, 5, 11)

    plot_goals_on_maze(
        maze, tasks_goals_fixed,
        title=f"(a) {NUM_TASKS} fixed goal points (Task 1..{NUM_TASKS})",
        grid=True, grid_color="0.7", grid_lw=0.6
    )

    print("\n==============================")
    print(f"Start δ={DELTA} (fixed), RUNS={RUNS}, record tasks={RECORD_TASKS}")
    print("Reward design: goal_reward=+1.0, step_cost=-0.01, wall_penalty=-0.05")
    print(f"Boltzmann grid: tau0 fixed = {TAU0_FIXED}, delta_tau in {DELTA_TAU_LIST}")
    print("==============================")

    # ---- Precompute seeds per run to make grid-search fair ----
    run_env_seeds = []
    run_prq_seeds = []
    for r in range(RUNS):
        rng = np.random.default_rng(2000 + r)
        env_seeds = rng.integers(0, 10**9, size=NUM_TASKS, dtype=np.int64)
        prq_seeds = rng.integers(0, 10**9, size=NUM_TASKS, dtype=np.int64)
        run_env_seeds.append(env_seeds)
        run_prq_seeds.append(prq_seeds)

    def baseline_seed(r, t1):
        return 9999 + 100 * r + int(t1)

    # ---- Collect baseline curves (shared across delta_tau grid) ----
    rl_curves_all = {t: [] for t in RECORD_TASKS}
    for r in range(RUNS):
        for t1 in RECORD_TASKS:
            goal = tasks_goals_fixed[t1 - 1]
            env = MazeEnv(maze, goal=goal, gamma=gamma, seed=int(run_env_seeds[r][t1 - 1]))
            _, curve_rl = q_learning_baseline_curve(
                env,
                K=K, Hmax=Hmax,
                alpha=alpha, gamma=gamma,
                eps_start=eps_start, eps_end=eps_end,
                seed=baseline_seed(r, t1)
            )
            rl_curves_all[t1].append(curve_rl)

    # ---- Collect PRQ curves for each delta_tau ----
    prq_curves_all = {dt: {t: [] for t in RECORD_TASKS} for dt in DELTA_TAU_LIST}

    for dt in DELTA_TAU_LIST:
        for r in range(RUNS):
            _, _, _, _, curves_dict, _goals_dict = plpr_run_boltzmann(
                tasks_goals_fixed,
                delta=DELTA,
                K=K, Hmax=Hmax,
                alpha=alpha, gamma=gamma,
                psi=psi, nu=nu,
                tau0=float(TAU0_FIXED),
                delta_tau=float(dt),
                env_seeds=run_env_seeds[r],
                prq_seeds=run_prq_seeds[r],
                run_label=f"[run {r+1:02d}/{RUNS:02d}]",
                record_task_indices_1based=RECORD_TASKS
            )
            for t1 in RECORD_TASKS:
                if t1 in curves_dict and curves_dict[t1] is not None:
                    prq_curves_all[dt][t1].append(curves_dict[t1])

    # ---- Plot: all delta_tau variants vs baseline (Taskごと) ----
    for t1 in RECORD_TASKS:
        if len(rl_curves_all[t1]) == 0:
            print(f"[WARN] no baseline curves for Task {t1}")
            continue

        prq_curves_by_label = {}
        for dt in DELTA_TAU_LIST:
            curves = prq_curves_all[dt][t1]
            if len(curves) > 0:
                prq_curves_by_label[f"PRQ Boltz (tau0={TAU0_FIXED}, dTau={dt})"] = curves

        if len(prq_curves_by_label) == 0:
            print(f"[WARN] no PRQ curves collected for Task {t1}")
            continue

        goal = tasks_goals_fixed[t1 - 1]
        plot_compare_multi_paper_like(
            prq_curves_by_label,
            rl_curves_all[t1],
            title=f"PRQ(Boltzmann dTau grid; tau0={TAU0_FIXED}) + Baseline - "
                  f"Task {t1}/{NUM_TASKS} goal={goal} (δ={DELTA}, runs={RUNS})",
            tick_every=200
        )

if __name__ == "__main__":
    main()
