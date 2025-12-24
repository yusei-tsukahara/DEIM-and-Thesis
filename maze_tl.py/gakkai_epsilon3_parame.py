import numpy as np
import random
import matplotlib.pyplot as plt
import pygame  # 迷路プレビュー用（任意）

# =====================
# 迷路表示（PyGame）
# =====================
def show_maze(maze, start, goal, cell_size=30, title="Maze Preview"):
    pygame.init()
    H, W = maze.shape
    screen = pygame.display.set_mode((W*cell_size, H*cell_size))
    pygame.display.set_caption(title)

    WHITE=(255,255,255); BLACK=(0,0,0); GREEN=(0,200,0); BLUE=(0,0,255)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)
        for y in range(H):
            for x in range(W):
                rect = pygame.Rect(x*cell_size, y*cell_size, cell_size, cell_size)
                if maze[y,x] == 1:
                    pygame.draw.rect(screen, BLACK, rect)
                elif (y,x) == goal:
                    pygame.draw.rect(screen, GREEN, rect)
                elif (y,x) == start:
                    pygame.draw.rect(screen, BLUE, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)  # 枠線
        pygame.display.flip()

    pygame.quit()

# ===== 再現性 =====
np.random.seed(42)
random.seed(42)

# ===== 迷路（1=壁, 0=道） =====
maze_A = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1,1,1,1,1],
    [1,0,1,1,0,1,0,0,0,0,0,1],
    [1,0,0,1,0,1,1,0,0,1,1,1],
    [1,0,0,1,0,1,1,0,1,1,0,1],
    [1,1,0,1,0,0,1,0,0,0,0,1],
    [1,0,0,1,1,0,1,1,1,0,1,1],
    [1,0,0,0,1,0,0,0,1,0,0,1],
    [1,0,0,1,1,1,1,0,0,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1]
])
maze_B = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1,1,1,1,1],
    [1,0,1,1,0,1,0,0,0,0,0,1],
    [1,0,0,1,0,1,1,0,0,1,1,1],
    [1,0,0,1,0,1,1,0,1,1,0,1],
    [1,1,0,1,0,0,1,0,0,0,0,1],
    [1,0,0,1,1,0,1,1,1,0,1,1],
    [1,0,0,0,1,0,0,0,1,0,0,1],
    [1,0,0,1,1,1,1,0,0,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1]
])

# ===== スタート/ゴール（y,x） =====
start_A = (1, 1);  goal_A = (10, 10)
start_B = (1, 1);  goal_B = (8, 3)  # (x,y)=(8,9)→(y,x)

# ===== 行動（dy,dx）上右下左 =====
actions = [(-1,0), (0,1), (1,0), (0,-1)]

# ===== ハイパーパラメータ =====
alpha = 0.5
gamma = 0.99
# εパラメータはsoftmaxでは未使用（比較用に保持）
epsilon_start = 1.0
epsilon_decay = 0.95
epsilon_min   = 0.01
episodes_A = 100
episodes_B = 100
step_limit  = 500
reward_goal = 10
reward_wall = -1
reward_step = -0.1

# ===== 行動選択：Boltzmann(softmax) 用 温度スケジュール =====
selection_mode = "softmax"  # ← デフォルトをsoftmaxへ
temp_start = 3.0
temp_min   = 0.5
temp_decay = 0.97

def get_temperature(ep: int) -> float:
    return max(temp_min, temp_start * (temp_decay ** ep))

# ===== Online-τ（自律部分転移）パラメータ（保守化） =====
tau_init = 0.02
tau_max  = 1.0
k_down   = 0.15
k_up     = 0.001
use_ema  = True
ema_beta = 0.75
margin   = 0.02
warmup_eps = 5

# ===== 共通関数 =====
def choose_action(state, Q, mode, temperature=None, epsilon=None):
    y, x = state
    q = Q[y, x]
    if mode == "softmax":
        T = max(1e-6, temperature if temperature is not None else 1.0)
        qv = np.clip(q, -500, 500)
        logits = (qv - np.max(qv)) / T
        exp_q = np.exp(logits)
        probs = exp_q / np.sum(exp_q)
        return int(np.random.choice(len(actions), p=probs))
    else:  # epsilon-greedy（互換用）
        if epsilon is None:
            epsilon = 0.1
        if random.random() < epsilon:
            return random.randint(0, len(actions)-1)
        return int(np.argmax(q))


def train_on_maze(maze, start, goal, episodes, Q_init=None):
    H, W = maze.shape
    Q = np.zeros((H, W, len(actions))) if Q_init is None else Q_init.copy()
    epsilon = epsilon_start
    steps_hist, rewards_hist = [], []

    for ep in range(episodes):
        state = start
        total_reward, steps, done = 0.0, 0, False
        T = get_temperature(ep)
        while not done and steps < step_limit:
            a_idx = choose_action(state, Q, selection_mode, temperature=T, epsilon=epsilon)
            dy, dx = actions[a_idx]
            y, x = state
            ny, nx = y+dy, x+dx

            if not (0 <= ny < H and 0 <= nx < W) or maze[ny, nx]==1:
                reward = reward_wall; next_state = state; terminal = False
            elif (ny, nx) == goal:
                reward = reward_goal; next_state = (ny, nx); terminal = True; done = True
            else:
                reward = reward_step; next_state = (ny, nx); terminal = False

            target = reward if terminal else reward + gamma*np.max(Q[next_state[0], next_state[1]])
            delta  = target - Q[y, x, a_idx]
            Q[y, x, a_idx] += alpha * delta

            state = next_state
            total_reward += reward
            steps += 1

        steps_hist.append(steps)
        rewards_hist.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q, steps_hist, rewards_hist

# =====================
# 自律部分転移（online-τ）
# =====================
def _normalize_Q_source(Q_source):
    """状態ごとに正規化：maxを0基準、範囲でスケーリング、tanhで飽和抑制"""
    Qs = Q_source.copy().astype(np.float64)
    mx = Qs.max(axis=2, keepdims=True)
    mn = Qs.min(axis=2, keepdims=True)
    rng = np.maximum(mx - mn, 1e-6)
    Qs = (Qs - mx) / rng          # 値域 ~ [-1, 0]
    Qs = np.tanh(Qs * 3.0)        # 外れ値を丸める
    return Qs


def train_online_tau(
    maze, start, goal, episodes, Q_source,
    tau_init=tau_init, tau_max=tau_max, k_down=k_down, k_up=k_up,
    use_ema=use_ema, ema_beta=ema_beta, margin=margin, warmup_eps=warmup_eps,
    use_norm=True
):
    """自律部分転移：Qc = Q_native + τ(s,a)*Q_source
    - τはTD誤差(EMA)でオンライン更新
    - 微小誤差はmargin内で無視
    - 前半warmup期間は転移を凍結
    - Q_sourceは状態ごとに正規化（既定）
    """
    H, W, A = Q_source.shape
    Q_native = np.zeros_like(Q_source, dtype=np.float64)
    tau_map  = np.full((H, W, A), float(tau_init))
    ema_map  = np.zeros((H, W, A), dtype=np.float64)

    # 正規化したソースQ
    Qs = _normalize_Q_source(Q_source) if use_norm else Q_source.astype(np.float64)

    epsilon = epsilon_start
    steps_hist, rewards_hist = [], []

    for ep in range(episodes):
        y, x = start
        total_reward, steps, done = 0.0, 0, False
        active_tau = (ep >= warmup_eps)
        T = get_temperature(ep)

        while not done and steps < step_limit:
            # 合成Q
            if active_tau:
                Qc_state = Q_native[y, x] + tau_map[y, x] * Qs[y, x]
            else:
                Qc_state = Q_native[y, x]

            # 行動選択：softmax（または互換用にepsilon-greedy）
            if selection_mode == "softmax":
                qv = np.clip(Qc_state, -500, 500)
                logits = (qv - np.max(qv)) / max(1e-6, T)
                exp_q = np.exp(logits)
                probs = exp_q / np.sum(exp_q)
                a_idx = int(np.random.choice(A, p=probs))
            else:
                if random.random() < epsilon:
                    a_idx = random.randint(0, A-1)
                else:
                    a_idx = int(np.argmax(Qc_state))

            dy, dx = actions[a_idx]
            ny, nx = y+dy, x+dx

            if not (0 <= ny < H and 0 <= nx < W) or maze[ny, nx]==1:
                reward = reward_wall; ny, nx = y, x; terminal = False
            elif (ny, nx) == goal:
                reward = reward_goal; terminal = True; done = True
            else:
                reward = reward_step; terminal = False

            if active_tau:
                Qc_next = Q_native[ny, nx] + tau_map[ny, nx] * Qs[ny, nx]
            else:
                Qc_next = Q_native[ny, nx]

            target  = reward if terminal else reward + gamma * np.max(Qc_next)

            Qc_sa = (Q_native[y, x, a_idx] + tau_map[y, x, a_idx] * Qs[y, x, a_idx]) if active_tau else Q_native[y, x, a_idx]
            delta = target - Qc_sa

            # ネイティブQの更新
            Q_native[y, x, a_idx] += alpha * delta

            # τ更新（warmup中はスキップ）
            if active_tau:
                d_eff = delta
                if use_ema:
                    ema_map[y, x, a_idx] = ema_beta * ema_map[y, x, a_idx] + (1-ema_beta) * delta
                    d_eff = ema_map[y, x, a_idx]

                if d_eff < -margin:
                    tau_map[y, x, a_idx] = max(0.0, tau_map[y, x, a_idx] + k_down * d_eff)
                elif d_eff > margin:
                    tau_map[y, x, a_idx] = min(tau_max, tau_map[y, x, a_idx] + k_up * d_eff)

            y, x = ny, nx
            total_reward += reward
            steps += 1

        steps_hist.append(steps)
        rewards_hist.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    Q_combined = Q_native + tau_map * Qs  # 参照用
    return Q_combined, steps_hist, rewards_hist

# ===== 可視化ユーティリティ =====
def band_plot(ylist, label):
    Y = np.array(ylist)
    m = Y.mean(axis=0)
    s = Y.std(axis=0)
    x = np.arange(len(m))
    plt.plot(x, m, label=label)
    plt.fill_between(x, m-s, m+s, alpha=0.2)


# ===== 実験フロー =====
if __name__ == "__main__":
    # 迷路プレビュー（任意。ウィンドウが出るので必要に応じてコメントアウト）
    # show_maze(maze_A, start_A, goal_A, title="Maze A")
    # show_maze(maze_B, start_B, goal_B, title="Maze B")

    # --- 1) Maze AでソースQを作成（単発）
    Q_A, _, _ = train_on_maze(maze_A, start_A, goal_A, episodes_A)

    # --- 2) Maze B：3手法（単発）
    Q_B_none, steps_B_none, rewards_B_none = train_on_maze(maze_B, start_B, goal_B, episodes_B)
    Q_B_simple, steps_B_simple, rewards_B_simple = train_on_maze(maze_B, start_B, goal_B, episodes_B, Q_init=Q_A)
    Q_B_onlinetau, steps_B_onlinetau, rewards_B_onlinetau = train_online_tau(
        maze_B, start_B, goal_B, episodes_B, Q_source=Q_A,
        tau_init=tau_init, tau_max=tau_max, k_down=k_down, k_up=k_up,
        use_ema=use_ema, ema_beta=ema_beta, margin=margin, warmup_eps=warmup_eps,
        use_norm=True
    )

    # --- 3) 学習曲線（単発表示）
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(steps_B_none,      label="B: no transfer")
    plt.plot(steps_B_simple,    label="B: simple copy")
    plt.plot(steps_B_onlinetau, label="B: online-τ (self selective)")
    plt.title("Steps per Episode (Maze B)")
    plt.xlabel("Episode"); plt.ylabel("Steps"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(rewards_B_none,      label="B: no transfer")
    plt.plot(rewards_B_simple,    label="B: simple copy")
    plt.plot(rewards_B_onlinetau, label="B: online-τ (self selective)")
    plt.title("Total Reward per Episode (Maze B)")
    plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.legend()

    plt.tight_layout(); plt.show()

    # --- 4) 複数seedで平均±標準偏差
    seeds = 10
    steps_none, steps_copy, steps_tau = [], [], []

    for s in range(seeds):
        np.random.seed(1000 + s)
        random.seed(1000 + s)

        # Aでソース学習
        Q_A_s, _, _ = train_on_maze(maze_A, start_A, goal_A, episodes_A)

        # 3手法（B）
        _, st_n, _ = train_on_maze(maze_B, start_B, goal_B, episodes_B)
        _, st_c, _ = train_on_maze(maze_B, start_B, goal_B, episodes_B, Q_init=Q_A_s)
        _, st_t, _ = train_online_tau(
            maze_B, start_B, goal_B, episodes_B, Q_source=Q_A_s,
            tau_init=tau_init, tau_max=tau_max, k_down=k_down, k_up=k_up,
            use_ema=use_ema, ema_beta=ema_beta, margin=margin, warmup_eps=warmup_eps,
            use_norm=True
        )

        steps_none.append(st_n)
        steps_copy.append(st_c)
        steps_tau.append(st_t)

    plt.figure(figsize=(12,5))
    band_plot(steps_none, 'no transfer')
    band_plot(steps_copy, 'simple copy')
    band_plot(steps_tau,  'online-τ')
    plt.title('Steps per Episode (mean ± std across seeds) — softmax policy')
    plt.xlabel('Episode'); plt.ylabel('Steps'); plt.legend(); plt.tight_layout(); plt.show()
