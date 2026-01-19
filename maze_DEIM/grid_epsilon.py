import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 0) Reward parameters (sparse: goal only)
# ============================================================
GOAL_REWARD = 1.0  # only reward when reaching goal

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
# 3) Environment (sparse reward: goal only)
# ============================================================
class MazeEnv:
    """
    Sparse reward design:
      - If reach goal: reward = goal_reward
      - Else: reward = 0
    Blocked moves simply keep the position (no penalty).
    """
    def __init__(self, maze_, goal, gamma=0.95, seed=0,
                 goal_reward=GOAL_REWARD):
        self.maze = maze_
        self.H, self.W = maze_.shape
        self.goal = tuple(goal)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)

        self.goal_reward = float(goal_reward)

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
        dx, dy = self.moves[int(a)]
        nx, ny = x + dx, y + dy

        # blocked => stay (no penalty)
        if (not self.in_bounds(nx, ny)) or self.is_wall(nx, ny):
            nx, ny = x, y

        self.pos = (nx, ny)
        done = (self.pos == self.goal)

        r = self.goal_reward if done else 0.0
        return self.state_id(self.pos), float(r), bool(done)

# ============================================================
# 4) Action selection helpers (paper ε-greedy = greedy probability)
# ============================================================
def greedy_action(Q, s, rng):
    qs = Q[s]
    mx = np.max(qs)
    cand = np.flatnonzero(qs == mx)
    return int(rng.choice(cand))

def random_action(n_actions, rng):
    return int(rng.integers(0, n_actions))

def eps1_greedy_action_paper(Q, s, eps1_greedy_prob, rng):
    """
    ★論文定義の ε1-greedy：
      - with prob ε1: greedy
      - with prob 1-ε1: random
    """
    eps1 = float(np.clip(eps1_greedy_prob, 0.0, 1.0))
    if rng.random() < eps1:
        return greedy_action(Q, s, rng)
    return random_action(Q.shape[1], rng)

def softmax_stable(values, tau):
    """
    Paper-style softmax uses exp(tau * W) (not exp(W/tau)).
    """
    v = np.asarray(values, dtype=float)
    z = float(tau) * v
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

# ============================================================
# 4.5) Bandit selection helpers
#   - (kept) Boltzmann / softmax
#   - (added) epsilon-greedy  ★方策選択をこちらに変更
# ============================================================
def bandit_select_arm_boltzmann(W, rng, tau=1.0):
    W = np.asarray(W, dtype=float)
    p = softmax_stable(W, tau)
    return int(rng.choice(np.arange(W.shape[0]), p=p))

def bandit_select_arm_eps_greedy(W, rng, eps=0.1):
    """
    ε-greedy for bandit arm selection over W:
      - with prob (1-ε): choose argmax W (tie-break random)
      - with prob ε:     choose random arm
    """
    W = np.asarray(W, dtype=float)
    eps = float(np.clip(eps, 0.0, 1.0))

    if rng.random() < eps:
        return int(rng.integers(0, W.shape[0]))

    mx = np.max(W)
    cand = np.flatnonzero(W == mx)
    return int(rng.choice(cand))

# ============================================================
# 5) Paper Gain (Eq.(1) style)
#   W = Σ_{h=0..Hmax-1} γ^h r_h   (no division by Hmax)
#   ★今回は r_eval = r_train（= 1 at goal, else 0）に統一
# ============================================================
def episode_gain_paper(r_eval_list, gamma, Hmax):
    G = 0.0
    disc = 1.0  # gamma^0
    for j in range(min(Hmax, len(r_eval_list))):
        G += disc * float(r_eval_list[j])
        disc *= float(gamma)
    return G

# ============================================================
# 6) One-episode routines
#   - 学習(Q更新)も評価も同じ報酬（goal=1 else 0）
# ============================================================
def q_learning_episode_greedy_paper_gain(env, Q, alpha, gamma, Hmax, rng):
    s = env.reset()
    r_eval_steps = []
    for _h in range(Hmax):
        a = greedy_action(Q, s, rng)
        ns, r_train, done = env.step(a)

        Q[s, a] += float(alpha) * ((r_train + float(gamma) * np.max(Q[ns])) - Q[s, a])

        r_eval_steps.append(float(r_train))
        s = ns
        if done:
            break

    return episode_gain_paper(r_eval_steps, gamma, Hmax)

def q_learning_episode_eps1_paper_gain(env, Q, alpha, gamma, Hmax, eps1, rng):
    s = env.reset()
    r_eval_steps = []
    for _h in range(Hmax):
        a = eps1_greedy_action_paper(Q, s, eps1, rng)
        ns, r_train, done = env.step(a)

        Q[s, a] += float(alpha) * ((r_train + float(gamma) * np.max(Q[ns])) - Q[s, a])

        r_eval_steps.append(float(r_train))
        s = ns
        if done:
            break

    return episode_gain_paper(r_eval_steps, gamma, Hmax)

def q_learning_episode_pi_reuse_paper_gain(env, Q_new, Q_past, alpha, gamma, Hmax,
                                          psi=1.0, nu=0.95, rng=None):
    """
    ★論文の実験設定に合わせた π-reuse：
      - with prob ψ_h:    Πpast(s)=greedy(Q_past)
      - else:             ε_h-greedy(Πnew) where ε_h = 1 - ψ_h （εはgreedy確率）
      - ψ_{h+1} = ν ψ_h   （stepごと減衰）
    Return: paper gain (Eq.(1) style)
    """
    if rng is None:
        raise ValueError("rng must be provided")

    s = env.reset()
    r_eval_steps = []
    psi_h = float(psi)

    for _h in range(Hmax):
        psi_h = float(np.clip(psi_h, 0.0, 1.0))

        if rng.random() < psi_h:
            a = greedy_action(Q_past, s, rng)
        else:
            eps_h = float(np.clip(1.0 - psi_h, 0.0, 1.0))
            a = eps1_greedy_action_paper(Q_new, s, eps_h, rng)

        ns, r_train, done = env.step(a)

        Q_new[s, a] += float(alpha) * ((r_train + float(gamma) * np.max(Q_new[ns])) - Q_new[s, a])

        r_eval_steps.append(float(r_train))
        s = ns

        psi_h *= float(nu)
        if done:
            break

    return episode_gain_paper(r_eval_steps, gamma, Hmax)

# ============================================================
# 7) Baseline curve (paper-like exploration schedule)
#   ε1 starts at 0 and increases by eps1_inc each episode
# ============================================================
def baseline_curve_eps1_increasing(env,
                                   K=2000, Hmax=100,
                                   alpha=0.05, gamma=0.95,
                                   eps1_start=0.0, eps1_inc=0.0005,
                                   seed=0,
                                   record_curve=True):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions), dtype=float)
    curve = np.zeros(K, dtype=float) if record_curve else None

    eps1 = float(eps1_start)
    for k in range(K):
        g = q_learning_episode_eps1_paper_gain(env, Q, alpha, gamma, Hmax, eps1, rng)
        if record_curve:
            curve[k] = g
        eps1 = float(np.clip(eps1 + eps1_inc, 0.0, 1.0))

    return Q, curve

# ============================================================
# 7.5) First task without reuse (paper PLPR behavior)
# ============================================================
def learn_first_task_without_reuse(env,
                                  K=2000, Hmax=100,
                                  alpha=0.05, gamma=0.95,
                                  eps1_start=0.0, eps1_inc=0.0005,
                                  seed=0,
                                  record_curve=False):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions), dtype=float)
    curve = np.zeros(K, dtype=float) if record_curve else None

    eps1 = float(eps1_start)
    total = 0.0
    for k in range(K):
        g = q_learning_episode_eps1_paper_gain(env, Q, alpha, gamma, Hmax, eps1, rng)
        total += g
        if record_curve:
            curve[k] = g
        eps1 = float(np.clip(eps1 + eps1_inc, 0.0, 1.0))

    W_omega = total / float(K)
    return Q, W_omega, curve

# ============================================================
# 8) PRQ-Learning (bandit = epsilon-greedy over W)  ★変更点
# ============================================================
def prq_learning_boltzmann(env, library_Q,
                           K=2000, Hmax=100,
                           alpha=0.05, gamma=0.95,
                           psi=1.0, nu=0.95,
                           tau0=0.0, delta_tau=0.05,
                           seed=0,
                           record_curve=False,
                           bandit_eps=0.1):
    rng = np.random.default_rng(seed)
    n = len(library_Q)

    Q_new = np.zeros((env.n_states, env.n_actions), dtype=float)
    W = np.zeros(n + 1, dtype=float)   # running mean gain per arm
    U = np.zeros(n + 1, dtype=int)     # counts per arm

    # tau は残す（他は変えない）が、ε-greedyでは使わない
    tau = float(tau0)

    total_g = 0.0
    curve = np.zeros(K, dtype=float) if record_curve else None

    for k in range(K):
        # ★方策（アーム）選択：Boltzmann -> ε-greedy
        chosen = bandit_select_arm_eps_greedy(W, rng, eps=bandit_eps)

        if chosen == 0:
            g = q_learning_episode_greedy_paper_gain(env, Q_new, alpha, gamma, Hmax, rng)
        else:
            g = q_learning_episode_pi_reuse_paper_gain(
                env, Q_new, library_Q[chosen - 1],
                alpha, gamma, Hmax,
                psi=psi, nu=nu, rng=rng
            )

        W[chosen] = (W[chosen] * U[chosen] + g) / (U[chosen] + 1)
        U[chosen] += 1

        total_g += g
        if record_curve:
            curve[k] = g

        # 残す（未使用）
        tau += float(delta_tau)

    avg_gain_overall = total_g / float(K)
    return Q_new, W, U, avg_gain_overall, curve

# ============================================================
# 9) PLPR (same criterion)
#   - Add criterion: Wmax < δ * WΩ
# ============================================================
def plpr_run_boltzmann(tasks_goals,
                       delta=0.25,
                       K=2000, Hmax=100,
                       alpha=0.05, gamma=0.95,
                       psi=1.0, nu=0.95,
                       tau0=0.0,
                       delta_tau=0.05,
                       env_seeds=None,
                       prq_seeds=None,
                       scratch_seeds=None,
                       run_label="",
                       record_task_indices_1based=(1, 5, 9, 12, 15),
                       eps1_start=0.0,
                       eps1_inc=0.0005,
                       bandit_eps=0.1):
    num_tasks = len(tasks_goals)
    if env_seeds is None or prq_seeds is None or scratch_seeds is None:
        raise ValueError("Please provide env_seeds, prq_seeds, scratch_seeds for fair runs.")
    if len(env_seeds) != num_tasks or len(prq_seeds) != num_tasks or len(scratch_seeds) != num_tasks:
        raise ValueError("env_seeds/prq_seeds/scratch_seeds must have length = num_tasks.")

    library = []
    gains = []
    lib_sizes = []
    added_flags = []
    core_goal_indices = []

    record_set0 = {i - 1 for i in record_task_indices_1based}
    curves_dict = {}
    goals_of_recorded = {}

    for t, goal in enumerate(tasks_goals):
        env = MazeEnv(maze, goal=goal, gamma=gamma, seed=int(env_seeds[t]),
                      goal_reward=GOAL_REWARD)
        record_this = (t in record_set0)

        if len(library) == 0:
            # FIRST task: learn WITHOUT reuse then add
            Q_new, W_omega, curve = learn_first_task_without_reuse(
                env,
                K=K, Hmax=Hmax,
                alpha=alpha, gamma=gamma,
                eps1_start=eps1_start, eps1_inc=eps1_inc,
                seed=int(scratch_seeds[t]),
                record_curve=record_this
            )

            library.append(Q_new)
            core_goal_indices.append(t)
            add = True
            W_max = float("-inf")
            avg_gain = float(W_omega)

            if record_this:
                idx1 = t + 1
                curves_dict[idx1] = curve
                goals_of_recorded[idx1] = goal

            gains.append(avg_gain)
            lib_sizes.append(len(library))
            added_flags.append(add)

            print(f"{run_label} [FIRST:no reuse] (tau0={tau0:.3f}, dTau={delta_tau:.3f}) "
                  f"Task {t+1:02d}/{num_tasks} |L|={len(library):02d} add={add} "
                  f"WΩ={W_omega:.6f} Wmax={W_max} avgGain={avg_gain:.6f} goal={goal}")
            continue

        # Subsequent tasks: PRQ-learning (now uses epsilon-greedy bandit)
        Q_new, W, U, avg_gain, curve = prq_learning_boltzmann(
            env, library,
            K=K, Hmax=Hmax,
            alpha=alpha, gamma=gamma,
            psi=psi, nu=nu,
            tau0=float(tau0), delta_tau=float(delta_tau),
            seed=int(prq_seeds[t]),
            record_curve=record_this,
            bandit_eps=float(bandit_eps)
        )

        if record_this:
            idx1 = t + 1
            curves_dict[idx1] = curve
            goals_of_recorded[idx1] = goal

        W_omega = float(W[0])
        W_max = float(np.max(W[1:]))

        add = (W_max < float(delta) * W_omega)
        if add:
            library.append(Q_new)
            core_goal_indices.append(t)

        gains.append(float(avg_gain))
        lib_sizes.append(len(library))
        added_flags.append(bool(add))

        print(f"{run_label} (tau0={tau0:.3f}, dTau={delta_tau:.3f}) "
              f"Task {t+1:02d}/{num_tasks} |L|={len(library):02d} add={add} "
              f"WΩ={W_omega:.6f} Wmax={W_max:.6f} avgGain={avg_gain:.6f} goal={goal}")

    return (np.array(gains), np.array(lib_sizes), np.array(added_flags),
            core_goal_indices, curves_dict, goals_of_recorded)

# ============================================================
# 10) Visualization (paper-like: cumulative mean gain)
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
                 label="Baseline Q-learning (paper ε1-greedy increasing)")

    plt.title(title)
    plt.xlabel("Trials")
    plt.ylabel("Cumulative mean gain (paper Eq.(1) gain)")
    plt.grid(True)
    plt.legend()
    plt.show()

# ============================================================
# 11) Main
# ============================================================
def main():
    DELTA = 0.25
    RUNS = 1

    K = 2000
    Hmax = 100
    gamma = 0.95
    alpha = 0.05

    psi = 1.0
    nu = 0.95

    # ★論文どおり固定（グリッドサーチ無し）
    TAU0_FIXED = 0.0
    DELTA_TAU = 0.05

    # ★方策（アーム）選択を ε-greedy に変更したので、そのε
    BANDIT_EPS = 0.10

    # Paper-like baseline/first-task exploration schedule (ε1 = greedy prob)
    EPS1_START = 0.0
    EPS1_INC = 0.0005

    NUM_TASKS = len(tasks_goals_fixed)
    RECORD_TASKS = (1, 5, 9, 12, 15)

    plot_goals_on_maze(
        maze, tasks_goals_fixed,
        title=f"(a) {NUM_TASKS} fixed goal points (Task 1..{NUM_TASKS})",
        grid=True, grid_color="0.7", grid_lw=0.6
    )

    print("\n==============================")
    print(f"Start δ={DELTA} (fixed), RUNS={RUNS}, record tasks={RECORD_TASKS}")
    print("Reward (sparse, unified): goal=+1 only, else 0 (no step cost / no wall penalty)")
    print("Evaluation/W update: paper gain Eq.(1) style (NO /H normalization; gamma^0 start)")
    print(f"Bandit policy selection: epsilon-greedy over W (bandit_eps={BANDIT_EPS})")
    print("π-reuse (paper experiment): within-episode ε_h = 1 - ψ_h  (ε is greedy prob)")
    print(f"  psi={psi}, nu={nu}")
    print(f"Baseline/first-task (paper): ε1 starts {EPS1_START} and +{EPS1_INC}/episode")
    print(f"(Kept params) tau0={TAU0_FIXED}, delta_tau={DELTA_TAU}  (not used in ε-greedy bandit)")
    print("==============================")

    # ---- Precompute seeds per run (fair) ----
    run_env_seeds = []
    run_prq_seeds = []
    run_scratch_seeds = []
    for r in range(RUNS):
        rng = np.random.default_rng(2000 + r)
        env_seeds = rng.integers(0, 10**9, size=NUM_TASKS, dtype=np.int64)
        prq_seeds = rng.integers(0, 10**9, size=NUM_TASKS, dtype=np.int64)
        scratch_seeds = rng.integers(0, 10**9, size=NUM_TASKS, dtype=np.int64)
        run_env_seeds.append(env_seeds)
        run_prq_seeds.append(prq_seeds)
        run_scratch_seeds.append(scratch_seeds)

    def baseline_seed(r, t1):
        return 9999 + 100 * r + int(t1)

    # ---- Collect baseline curves (paper gain curves) ----
    rl_curves_all = {t: [] for t in RECORD_TASKS}
    for r in range(RUNS):
        for t1 in RECORD_TASKS:
            goal = tasks_goals_fixed[t1 - 1]
            env = MazeEnv(maze, goal=goal, gamma=gamma, seed=int(run_env_seeds[r][t1 - 1]))
            _, curve_rl = baseline_curve_eps1_increasing(
                env,
                K=K, Hmax=Hmax,
                alpha=alpha, gamma=gamma,
                eps1_start=EPS1_START, eps1_inc=EPS1_INC,
                seed=baseline_seed(r, t1),
                record_curve=True
            )
            rl_curves_all[t1].append(curve_rl)

    # ---- Collect PRQ curves ----
    prq_curves_all = {t: [] for t in RECORD_TASKS}

    for r in range(RUNS):
        _, _, _, _, curves_dict, _ = plpr_run_boltzmann(
            tasks_goals_fixed,
            delta=DELTA,
            K=K, Hmax=Hmax,
            alpha=alpha, gamma=gamma,
            psi=psi, nu=nu,
            tau0=float(TAU0_FIXED),
            delta_tau=float(DELTA_TAU),
            env_seeds=run_env_seeds[r],
            prq_seeds=run_prq_seeds[r],
            scratch_seeds=run_scratch_seeds[r],
            run_label=f"[run {r+1:02d}/{RUNS:02d}]",
            record_task_indices_1based=RECORD_TASKS,
            eps1_start=EPS1_START,
            eps1_inc=EPS1_INC,
            bandit_eps=float(BANDIT_EPS)
        )

        for t1 in RECORD_TASKS:
            if t1 in curves_dict and curves_dict[t1] is not None:
                prq_curves_all[t1].append(curves_dict[t1])

    # ---- Plot comparisons ----
    for t1 in RECORD_TASKS:
        if len(prq_curves_all[t1]) == 0:
            print(f"[WARN] no PRQ curves collected for Task {t1}")
            continue

        goal = tasks_goals_fixed[t1 - 1]
        prq_curves_by_label = {
            f"PLPR+PRQ bandit=eps-greedy (eps={BANDIT_EPS})": prq_curves_all[t1]
        }

        plot_compare_multi_paper_like(
            prq_curves_by_label,
            rl_curves_all[t1],
            title=f"PLPR+PRQ(eps-greedy bandit) vs Baseline - "
                  f"Task {t1}/{NUM_TASKS} goal={goal} (δ={DELTA}, runs={RUNS})",
            tick_every=200
        )

if __name__ == "__main__":
    main()
