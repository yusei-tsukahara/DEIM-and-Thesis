import numpy as np
import random
import matplotlib.pyplot as plt
import pygame  # ← 追加

# ===== 迷路表示（PyGame） =====
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
    [1,1,0,1,0,0,1,0,0,1,0,1],
    [1,0,0,1,1,0,1,1,1,0,1,1],
    [1,0,0,0,1,0,0,1,1,0,0,1],
    [1,0,1,1,1,1,1,0,0,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1]
])

# ===== スタート/ゴール（y,x） =====
start_A = (1, 1);  goal_A = (10, 10)
start_B = (1, 1);  goal_B = (10, 9)  # (x,y)=(8,9)→(y,x)

# ===== 行動（dy,dx）上右下左 =====
actions = [(-1,0), (0,1), (1,0), (0,-1)]

# ===== ハイパーパラメータ =====
alpha = 0.5
gamma = 0.99
epsilon_start = 1.0      # ← これを「温度τの初期値」としても使う
epsilon_decay = 0.90
epsilon_min   = 0.01
episodes_A = 100
episodes_B = 100
step_limit  = 500
reward_goal = 10
reward_wall = -1
reward_step = -0.1

# ===== Online-τ（自律部分転移）パラメータ =====
tau_init = 0.20
tau_max  = 1.0
k_down   = 0.08   # δ<0 で素早く下げる
k_up     = 0.004  # δ>0 でゆっくり上げる
use_ema  = True
ema_beta = 0.85

# ===== Boltzmann (softmax) 行動選択 =====
def softmax_action(q_values, tau):
    """q_values: 1次元 (各行動のQ値), tau: 温度パラメータ"""
    # 過度なオーバーフロー防止
    qv = np.clip(q_values, -500, 500)
    # tau が 0 に近すぎると数値爆発するので下限を設ける
    tau = max(tau, 0.05)
    exp_q = np.exp((qv - np.max(qv)) / tau)
    probs = exp_q / np.sum(exp_q)
    return int(np.random.choice(len(qv), p=probs))

# ===== Maze A / B 共通のQ学習（行動選択＝Boltzmann） =====
def train_on_maze(maze, start, goal, episodes, Q_init=None):
    H, W = maze.shape
    Q = np.zeros((H, W, len(actions))) if Q_init is None else Q_init.copy()
    epsilon = epsilon_start  # ← これを「τスケジュール」として使う
    steps_hist, rewards_hist = [], []

    for ep in range(episodes):
        state = start
        total_reward, steps, done = 0.0, 0, False
        while not done and steps < step_limit:
            y, x = state
            # --- Boltzmann選択 ---
            tau = max(epsilon, 0.05)         # 温度（エピソードで徐々に冷やす）
            a_idx = softmax_action(Q[y, x], tau)

            dy, dx = actions[a_idx]
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
        # τに相当するepsilonを減衰させる（→だんだんgreedy寄り）
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q, steps_hist, rewards_hist

# ===== ユーザー版アルゴリズム：online-τ も Boltzmann に統一 =====
def train_online_tau(maze, start, goal, episodes, Q_source,
                     tau_init=0.35, tau_max=1.0, k_down=0.03, k_up=0.003,
                     use_ema=True, ema_beta=0.9):
    """自律部分転移：Q_native + τ(s,a)*Q_source、τをTD誤差でオンライン更新"""
    H, W, A = Q_source.shape
    Q_native = np.zeros_like(Q_source)
    tau_map  = np.full((H, W, A), tau_init)
    ema_map  = np.zeros((H, W, A), dtype=np.float64)

    epsilon = epsilon_start  # ← これを温度スケジュールに使う
    steps_hist, rewards_hist = [], []

    for ep in range(episodes):
        y, x = start
        total_reward, steps, done = 0.0, 0, False

        while not done and steps < step_limit:
            # 転移込みのQ値
            Qc = Q_native[y, x] + tau_map[y, x] * Q_source[y, x]

            # --- Boltzmann選択（online-τ版も同じルール） ---
            tau_temp = max(epsilon, 0.05)
            a_idx = softmax_action(Qc, tau_temp)

            dy, dx = actions[a_idx]
            ny, nx = y+dy, x+dx

            if not (0 <= ny < H and 0 <= nx < W) or maze[ny, nx]==1:
                reward = reward_wall; ny, nx = y, x; terminal = False
            elif (ny, nx) == goal:
                reward = reward_goal; terminal = True; done = True
            else:
                reward = reward_step; terminal = False

            Qc_next = Q_native[ny, nx] + tau_map[ny, nx] * Q_source[ny, nx]
            target  = reward if terminal else reward + gamma * np.max(Qc_next)

            Qc_sa = Q_native[y, x, a_idx] + tau_map[y, x, a_idx] * Q_source[y, x, a_idx]
            delta = target - Qc_sa

            # native Q の更新
            Q_native[y, x, a_idx] += alpha * delta

            # τ のオンライン更新
            d_eff = delta
            if use_ema:
                ema_map[y, x, a_idx] = ema_beta * ema_map[y, x, a_idx] + (1-ema_beta) * delta
                d_eff = ema_map[y, x, a_idx]

            if d_eff < 0:
                tau_map[y, x, a_idx] = max(0.0, tau_map[y, x, a_idx] + k_down * d_eff)  # d_eff<0
            elif d_eff > 0:
                tau_map[y, x, a_idx] = min(tau_max, tau_map[y, x, a_idx] + k_up * d_eff)

            y, x = ny, nx
            total_reward += reward
            steps += 1

        steps_hist.append(steps)
        rewards_hist.append(total_reward)
        # τに相当するepsilonを減衰
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    Q_combined = Q_native + tau_map * Q_source
    return Q_combined, steps_hist, rewards_hist

# ===== showcaseユーティリティ =====
def band_plot(ylist, label):
    Y = np.array(ylist)
    m = Y.mean(axis=0)
    s = Y.std(axis=0)
    x = np.arange(len(m))
    plt.plot(x, m, label=label)
    plt.fill_between(x, m-s, m+s, alpha=0.2)

# ===== 実験フロー =====
if __name__ == "__main__":
    # 迷路プレビュー（任意）
    show_maze(maze_A, start_A, goal_A, title="Maze A")
    show_maze(maze_B, start_B, goal_B, title="Maze B")

    # --- 1) Maze AでソースQを作成（単発）
    Q_A, _, _ = train_on_maze(maze_A, start_A, goal_A, episodes_A)

    # --- 2) Maze B：3手法（単発）
    Q_B_none,   steps_B_none,   rewards_B_none   = train_on_maze(maze_B, start_B, goal_B, episodes_B)
    Q_B_simple, steps_B_simple, rewards_B_simple = train_on_maze(maze_B, start_B, goal_B, episodes_B, Q_init=Q_A)
    Q_B_onlinetau, steps_B_onlinetau, rewards_B_onlinetau = train_online_tau(
        maze_B, start_B, goal_B, episodes_B, Q_source=Q_A,
        tau_init=tau_init, tau_max=tau_max, k_down=k_down, k_up=k_up,
        use_ema=use_ema, ema_beta=ema_beta
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

    # --- 4) 複数seedで平均±標準偏差のみを表示
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
            use_ema=use_ema, ema_beta=ema_beta
        )

        steps_none.append(st_n)
        steps_copy.append(st_c)
        steps_tau.append(st_t)

    plt.figure(figsize=(12,5))
    band_plot(steps_none, 'no transfer')
    band_plot(steps_copy, 'simple copy')
    band_plot(steps_tau,  'online-τ')
    plt.title('Steps per Episode (mean ± std across seeds) — Boltzmann policy')
    plt.xlabel('Episode'); plt.ylabel('Steps'); plt.legend(); plt.tight_layout(); plt.show()
