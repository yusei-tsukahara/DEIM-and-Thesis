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


def free_cells(maze_):
    ys, xs = np.where(maze_ == 0)
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
# 3) Environment  ★報酬設計（罰則あり）：goal_reward + step_cost (+ wall_penalty)
# ============================================================
class MazeEnv:
    """
    Reward design (dense / penalty):
      - Each step: step_cost (negative)
      - If action hits wall/out-of-bounds (no movement): add wall_penalty (negative)
      - If reach goal: reward = goal_reward (override step penalties for that step)
    """
    def __init__(self, maze_, goal, gamma=0.95, seed=0,
                 goal_reward=1.0, step_cost=-0.01, wall_penalty=-0.05):
        self.maze = maze_
        self.H, self.W = maze_.shape
        self.goal = tuple(goal)
        self.gamma = float(gamma)
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

        return self.state_id(self.pos), float(r), bool(done)

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

def softmax_stable(values, inv_temp):
    # inv_temp = "inverse temperature" style: p ∝ exp(inv_temp * value)
    v = np.asarray(values, dtype=float)
    z = inv_temp * v
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

# ============================================================
# 5) Bandit (policy-choice) selection: Boltzmann / ε-greedy / UCB
# ============================================================
def bandit_select_arm(W, rng, mode="boltzmann", inv_temp=0.0, eps_bandit=0.1,
                      U=None, t=None, c_ucb=1.0):
    """
    Choose chosen ∈ {0..n}:
      chosen==0 => use greedy on Q_new (omega)
      chosen>0  => reuse library_Q[chosen-1] with PRQ
    """
    W = np.asarray(W, dtype=float)
    n_arms = W.shape[0]

    if mode == "boltzmann":
        p = softmax_stable(W, inv_temp)
        return int(rng.choice(np.arange(n_arms), p=p))

    if mode == "eps_greedy":
        if rng.random() < eps_bandit:
            return int(rng.integers(0, n_arms))
        mx = np.max(W)
        cand = np.flatnonzero(W == mx)
        return int(rng.choice(cand))

    if mode == "ucb":
        if U is None or t is None:
            raise ValueError("UCB mode requires U (counts) and t (time).")
        U = np.asarray(U, dtype=float)

        untried = np.flatnonzero(U <= 0)
        if untried.size > 0:
            return int(rng.choice(untried))

        tt = max(1, int(t))
        bonus = float(c_ucb) * np.sqrt(np.log(tt) / U)
        scores = W + bonus
        mx = np.max(scores)
        cand = np.flatnonzero(scores == mx)
        return int(rng.choice(cand))

    raise ValueError(f"Unknown bandit mode: {mode}")

# ============================================================
# 6) One episode routines
# ============================================================
def q_learning_episode_greedy(env, Q, alpha, gamma, max_steps, rng):
    s = env.reset()
    G = 0.0
    disc = 1.0
    for _ in range(max_steps):
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
    for _ in range(max_steps):
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
            a = greedy_action(Q_past, s, rng)          # follow past policy
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
# 7) Baseline Q-learning runner (curve)
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
# 8) PRQ-Learning (policy-choice bandit)  + record raw G per episode
# ============================================================
def prq_learning(env, library_Q,
                 K=2000, Hmax=100,
                 alpha=0.05, gamma=0.95,
                 psi=1.0, nu=0.95,
                 inv_temp0=0.0, delta_inv_temp=0.05,
                 seed=0,
                 record_curve=False,
                 bandit_mode="boltzmann",
                 eps_bandit=0.1,
                 c_ucb=1.0):
    rng = np.random.default_rng(seed)
    n = len(library_Q)

    Q_new = np.zeros((env.n_states, env.n_actions), dtype=float)
    W = np.zeros(n + 1, dtype=float)   # running avg gain per arm
    U = np.zeros(n + 1, dtype=int)     # counts

    inv_temp = float(inv_temp0)
    total_G = 0.0
    curve = np.zeros(K, dtype=float) if record_curve else None

    for k in range(K):
        chosen = bandit_select_arm(
            W, rng,
            mode=bandit_mode,
            inv_temp=inv_temp,
            eps_bandit=eps_bandit,
            U=U, t=(k + 1), c_ucb=c_ucb
        )

        if chosen == 0:
            G = q_learning_episode_greedy(env, Q_new, alpha, gamma, Hmax, rng)
        else:
            G = q_learning_episode_pi_reuse(
                env, Q_new, library_Q[chosen - 1],
                alpha, gamma, Hmax, psi=psi, nu=nu, rng=rng
            )

        W[chosen] = (W[chosen] * U[chosen] + G) / (U[chosen] + 1)
        U[chosen] += 1

        # Boltzmannのみ「温度(逆温度)を上げていく」挙動を再現
        if bandit_mode == "boltzmann":
            inv_temp += float(delta_inv_temp)

        total_G += G
        if record_curve:
            curve[k] = G

    avg_gain_overall = total_G / K
    return Q_new, W, U, avg_gain_overall, curve

# ============================================================
# 9) PLPR (library growth with delta)
#    ★公平比較のため env_seeds/prq_seeds を外から渡せる
# ============================================================
def plpr_run(tasks_goals,
             delta=0.25,
             K=2000, Hmax=100,
             alpha=0.05, gamma=0.95,
             psi=1.0, nu=0.95,
             inv_temp0=0.0, delta_inv_temp=0.05,
             env_seeds=None,      # len=num_tasks
             prq_seeds=None,      # len=num_tasks
             run_label="",
             record_task_indices_1based=(3, 5, 11),
             bandit_mode="boltzmann",
             eps_bandit=0.1,
             c_ucb=1.0,
             # reward params
             goal_reward=1.0, step_cost=-0.01, wall_penalty=-0.05):
    num_tasks = len(tasks_goals)
    if env_seeds is None or prq_seeds is None:
        raise ValueError("Please provide env_seeds and prq_seeds for fair comparison.")
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
        env = MazeEnv(
            maze, goal=goal, gamma=gamma, seed=int(env_seeds[t]),
            goal_reward=goal_reward, step_cost=step_cost, wall_penalty=wall_penalty
        )
        record_this = (t in record_set0)

        Q_new, W, U, avg_gain, curve = prq_learning(
            env, library,
            K=K, Hmax=Hmax,
            alpha=alpha, gamma=gamma,
            psi=psi, nu=nu,
            inv_temp0=inv_temp0, delta_inv_temp=delta_inv_temp,
            seed=int(prq_seeds[t]),
            record_curve=record_this,
            bandit_mode=bandit_mode,
            eps_bandit=eps_bandit,
            c_ucb=c_ucb
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

        print(f"{run_label} {bandit_mode:9s} Task {t+1:02d}/{num_tasks} "
              f"|L|={len(library):02d} add={add} "
              f"WΩ={W_omega:.4f} Wmax={W_max:.4f} avgGain={avg_gain:.4f} goal={goal}")

    return (np.array(gains), np.array(lib_sizes), np.array(added_flags),
            core_goal_indices, curves_dict, goals_of_recorded)

# ============================================================
# 10) Visualization helpers
# ============================================================
def plot_goals_on_maze(maze_, goals, title="(a) goal points",
                       grid=True, grid_color="0.7", grid_lw=0.6):
    H_, W_ = maze_.shape
    fig, ax = plt.subplots(figsize=(7, 6))

    img = np.zeros((H_, W_, 3), dtype=float)
    img[maze_ == 0] = (1, 1, 1)
    img[maze_ == 1] = (0, 0, 0)

    ax.imshow(img, interpolation="none")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    if grid:
        ax.set_xticks(np.arange(-0.5, W_, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H_, 1), minor=True)
        ax.grid(which="minor", color=grid_color, linewidth=grid_lw)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xlim(-0.5, W_ - 0.5)
        ax.set_ylim(H_ - 0.5, -0.5)

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
        K_ = m.shape[0]
        xs = np.arange(tick_every, K_ + 1, tick_every)
        idx = xs - 1
        fmt = fmts[i % len(fmts)]
        plt.errorbar(xs, m[idx], yerr=s[idx], fmt=fmt, capsize=3, label=label)

    Xr = np.stack([cumulative_mean(c) for c in rl_curves], axis=0)
    mr, sr = Xr.mean(axis=0), Xr.std(axis=0)
    K_ = mr.shape[0]
    xs = np.arange(tick_every, K_ + 1, tick_every)
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
# 11) Grid-search -> choose BEST per bandit -> compare curves
#     ★罰則あり報酬設計での「最高性能比較」コード
# ============================================================
def main():
    # -----------------------------
    # Core experimental settings
    # -----------------------------
    DELTA = 0.25
    RUNS = 5
    K = 2000
    Hmax = 100
    gamma = 0.95
    alpha = 0.05

    # PRQ parameters
    psi = 1.0
    nu = 0.95

    # Reward design (dense / penalty)
    GOAL_REWARD = 1.0
    STEP_COST = -0.01
    WALL_PENALTY = -0.05

    # Baseline action-side exploration
    eps_start = 1.0
    eps_end = 0.05

    # Which tasks to plot curves for
    RECORD_TASKS = (3, 5, 11)

    NUM_TASKS = len(tasks_goals_fixed)

    # -----------------------------
    # Candidate grids (edit freely)
    # -----------------------------
    BOLTZ_GRID = [
        (0.00, 0.05),
        (0.05, 0.05),
        (0.10, 0.05),
        (0.05, 0.02),
    ]
    EPSB_GRID = [0.01, 0.05, 0.10, 0.20, 0.30]
    C_UCB_GRID = [0.03, 0.10, 0.30, 1.0, 3.0, 10.0]

    plot_goals_on_maze(
        maze, tasks_goals_fixed,
        title=f"(a) {NUM_TASKS} fixed goal points (Task 1..{NUM_TASKS})",
        grid=True, grid_color="0.7", grid_lw=0.6
    )

    print("\n==============================")
    print("Reward design (dense/penalty):")
    print(f"  goal_reward={GOAL_REWARD}, step_cost={STEP_COST}, wall_penalty={WALL_PENALTY}")
    print(f"δ={DELTA}, RUNS={RUNS}, K={K}, Hmax={Hmax}")
    print("==============================\n")

    # -----------------------------
    # Precompute seeds for fairness
    # -----------------------------
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

    # =========================================================
    # A) Grid search: pick BEST Boltzmann (inv_temp0, delta_inv_temp)
    # =========================================================
    boltz_scores = []
    for inv0, dinv in BOLTZ_GRID:
        per_run = []
        for r in range(RUNS):
            gains, *_ = plpr_run(
                tasks_goals_fixed,
                delta=DELTA,
                K=K, Hmax=Hmax,
                alpha=alpha, gamma=gamma,
                psi=psi, nu=nu,
                inv_temp0=inv0, delta_inv_temp=dinv,
                env_seeds=run_env_seeds[r],
                prq_seeds=run_prq_seeds[r],
                run_label=f"[GRID][run {r+1:02d}/{RUNS:02d}]",
                record_task_indices_1based=(),   # grid段階では曲線不要
                bandit_mode="boltzmann",
                eps_bandit=0.1,
                c_ucb=1.0,
                goal_reward=GOAL_REWARD, step_cost=STEP_COST, wall_penalty=WALL_PENALTY
            )
            per_run.append(float(np.mean(gains)))
        score = float(np.mean(per_run))
        boltz_scores.append((score, inv0, dinv))
        print(f"[GRID] Boltzmann inv0={inv0:.3f}, dinv={dinv:.3f} => meanScore={score:.6f}")

    boltz_scores.sort(key=lambda x: x[0], reverse=True)
    BEST_BOLTZ_SCORE, BEST_INV0, BEST_DINV = boltz_scores[0]
    print(f"\n[BEST] Boltzmann: inv0={BEST_INV0:.3f}, dinv={BEST_DINV:.3f}, score={BEST_BOLTZ_SCORE:.6f}\n")

    # =========================================================
    # B) Grid search: pick BEST ε-bandit (epsb)
    # =========================================================
    epsb_scores = []
    for epsb in EPSB_GRID:
        per_run = []
        for r in range(RUNS):
            gains, *_ = plpr_run(
                tasks_goals_fixed,
                delta=DELTA,
                K=K, Hmax=Hmax,
                alpha=alpha, gamma=gamma,
                psi=psi, nu=nu,
                inv_temp0=BEST_INV0, delta_inv_temp=BEST_DINV,  # 未使用だが引数は埋める
                env_seeds=run_env_seeds[r],
                prq_seeds=run_prq_seeds[r],
                run_label=f"[GRID][run {r+1:02d}/{RUNS:02d}]",
                record_task_indices_1based=(),
                bandit_mode="eps_greedy",
                eps_bandit=float(epsb),
                c_ucb=1.0,
                goal_reward=GOAL_REWARD, step_cost=STEP_COST, wall_penalty=WALL_PENALTY
            )
            per_run.append(float(np.mean(gains)))
        score = float(np.mean(per_run))
        epsb_scores.append((score, float(epsb)))
        print(f"[GRID] ε-bandit epsb={epsb:.3f} => meanScore={score:.6f}")

    epsb_scores.sort(key=lambda x: x[0], reverse=True)
    BEST_EPSB_SCORE, BEST_EPSB = epsb_scores[0]
    print(f"\n[BEST] ε-bandit: epsb={BEST_EPSB:.3f}, score={BEST_EPSB_SCORE:.6f}\n")

    # =========================================================
    # C) Grid search: pick BEST UCB (c)
    # =========================================================
    ucb_scores = []
    for c in C_UCB_GRID:
        per_run = []
        for r in range(RUNS):
            gains, *_ = plpr_run(
                tasks_goals_fixed,
                delta=DELTA,
                K=K, Hmax=Hmax,
                alpha=alpha, gamma=gamma,
                psi=psi, nu=nu,
                inv_temp0=BEST_INV0, delta_inv_temp=BEST_DINV,  # 未使用だが引数は埋める
                env_seeds=run_env_seeds[r],
                prq_seeds=run_prq_seeds[r],
                run_label=f"[GRID][run {r+1:02d}/{RUNS:02d}]",
                record_task_indices_1based=(),
                bandit_mode="ucb",
                eps_bandit=0.1,
                c_ucb=float(c),
                goal_reward=GOAL_REWARD, step_cost=STEP_COST, wall_penalty=WALL_PENALTY
            )
            per_run.append(float(np.mean(gains)))
        score = float(np.mean(per_run))
        ucb_scores.append((score, float(c)))
        print(f"[GRID] UCB c={c:.3f} => meanScore={score:.6f}")

    ucb_scores.sort(key=lambda x: x[0], reverse=True)
    BEST_UCB_SCORE, BEST_C = ucb_scores[0]
    print(f"\n[BEST] UCB: c={BEST_C:.3f}, score={BEST_UCB_SCORE:.6f}\n")

    # =========================================================
    # D) With BEST params: collect curves and compare
    # =========================================================
    print("\n==============================")
    print("COMPARE with BEST params (dense/penalty reward):")
    print(f"  Boltzmann: inv0={BEST_INV0}, dinv={BEST_DINV}")
    print(f"  ε-bandit : epsb={BEST_EPSB}")
    print(f"  UCB      : c={BEST_C}")
    print("==============================\n")

    # Baseline curves
    rl_curves_all = {t: [] for t in RECORD_TASKS}
    for r in range(RUNS):
        for t1 in RECORD_TASKS:
            goal = tasks_goals_fixed[t1 - 1]
            env = MazeEnv(
                maze, goal=goal, gamma=gamma, seed=int(run_env_seeds[r][t1 - 1]),
                goal_reward=GOAL_REWARD, step_cost=STEP_COST, wall_penalty=WALL_PENALTY
            )
            _, curve_rl = q_learning_baseline_curve(
                env,
                K=K, Hmax=Hmax,
                alpha=alpha, gamma=gamma,
                eps_start=eps_start, eps_end=eps_end,
                seed=baseline_seed(r, t1)
            )
            rl_curves_all[t1].append(curve_rl)

    # PRQ curves (3 bandits)
    prq_curves_all = {
        "PRQ bandit=Boltzmann (best)": {t: [] for t in RECORD_TASKS},
        "PRQ bandit=ε-greedy (best)":  {t: [] for t in RECORD_TASKS},
        "PRQ bandit=UCB (best)":       {t: [] for t in RECORD_TASKS},
    }

    for r in range(RUNS):
        # Boltzmann best
        _, _, _, _, curves_dict, _ = plpr_run(
            tasks_goals_fixed,
            delta=DELTA,
            K=K, Hmax=Hmax,
            alpha=alpha, gamma=gamma,
            psi=psi, nu=nu,
            inv_temp0=BEST_INV0, delta_inv_temp=BEST_DINV,
            env_seeds=run_env_seeds[r],
            prq_seeds=run_prq_seeds[r],
            run_label=f"[BEST][run {r+1:02d}/{RUNS:02d}]",
            record_task_indices_1based=RECORD_TASKS,
            bandit_mode="boltzmann",
            eps_bandit=BEST_EPSB,
            c_ucb=BEST_C,
            goal_reward=GOAL_REWARD, step_cost=STEP_COST, wall_penalty=WALL_PENALTY
        )
        for t1 in RECORD_TASKS:
            if t1 in curves_dict and curves_dict[t1] is not None:
                prq_curves_all["PRQ bandit=Boltzmann (best)"][t1].append(curves_dict[t1])

        # ε-bandit best
        _, _, _, _, curves_dict, _ = plpr_run(
            tasks_goals_fixed,
            delta=DELTA,
            K=K, Hmax=Hmax,
            alpha=alpha, gamma=gamma,
            psi=psi, nu=nu,
            inv_temp0=BEST_INV0, delta_inv_temp=BEST_DINV,
            env_seeds=run_env_seeds[r],
            prq_seeds=run_prq_seeds[r],
            run_label=f"[BEST][run {r+1:02d}/{RUNS:02d}]",
            record_task_indices_1based=RECORD_TASKS,
            bandit_mode="eps_greedy",
            eps_bandit=BEST_EPSB,
            c_ucb=BEST_C,
            goal_reward=GOAL_REWARD, step_cost=STEP_COST, wall_penalty=WALL_PENALTY
        )
        for t1 in RECORD_TASKS:
            if t1 in curves_dict and curves_dict[t1] is not None:
                prq_curves_all["PRQ bandit=ε-greedy (best)"][t1].append(curves_dict[t1])

        # UCB best
        _, _, _, _, curves_dict, _ = plpr_run(
            tasks_goals_fixed,
            delta=DELTA,
            K=K, Hmax=Hmax,
            alpha=alpha, gamma=gamma,
            psi=psi, nu=nu,
            inv_temp0=BEST_INV0, delta_inv_temp=BEST_DINV,
            env_seeds=run_env_seeds[r],
            prq_seeds=run_prq_seeds[r],
            run_label=f"[BEST][run {r+1:02d}/{RUNS:02d}]",
            record_task_indices_1based=RECORD_TASKS,
            bandit_mode="ucb",
            eps_bandit=BEST_EPSB,
            c_ucb=BEST_C,
            goal_reward=GOAL_REWARD, step_cost=STEP_COST, wall_penalty=WALL_PENALTY
        )
        for t1 in RECORD_TASKS:
            if t1 in curves_dict and curves_dict[t1] is not None:
                prq_curves_all["PRQ bandit=UCB (best)"][t1].append(curves_dict[t1])

    # Plot per task
    for t1 in RECORD_TASKS:
        prq_curves_by_label = {k: v[t1] for k, v in prq_curves_all.items() if len(v[t1]) > 0}
        if len(prq_curves_by_label) == 0:
            print(f"[WARN] no PRQ curves for Task {t1}")
            continue
        if len(rl_curves_all[t1]) == 0:
            print(f"[WARN] no baseline curves for Task {t1}")
            continue

        goal = tasks_goals_fixed[t1 - 1]
        plot_compare_multi_paper_like(
            prq_curves_by_label,
            rl_curves_all[t1],
            title=(f"(dense/penalty) BEST bandits vs Baseline - Task {t1}/{NUM_TASKS} "
                   f"goal={goal} (δ={DELTA}, runs={RUNS})"),
            tick_every=200
        )

if __name__ == "__main__":
    main()
