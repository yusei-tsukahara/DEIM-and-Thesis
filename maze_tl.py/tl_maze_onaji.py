import numpy as np
import random
import matplotlib.pyplot as plt
import pygame

# ===== 共通設定 =====
start = (1, 1)  # (x, y)
actions = [(0,-1),(1,0),(0,1),(-1,0)]  # 上・右・下・左（dx,dy）

# 学習パラメータ
alpha_init = 0.5
alpha_final = 0.05
gamma = 0.99
episodes = 100
EPS_END = 0.05      # εの下限（探索は少し残す）

# 報酬
reward_goal = 120.0
reward_wall = -10.0
reward_step = -1.0

# 実行まわり
STEP_LIMIT = 1000
CELL = 30
WHITE=(255,255,255); BLACK=(0,0,0); BLUE=(0,0,255); GREEN=(0,200,0)

# ===== 迷路A（転移元） =====
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
goal_A = (10,10)

# ===== 迷路B（転移先） =====
maze_B = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1,1,1,1,1],
    [1,0,1,1,0,1,0,0,0,0,0,1],
    [1,0,0,1,0,1,1,0,0,1,1,1],
    [1,0,0,1,0,1,1,0,1,1,0,1],
    [1,1,0,1,0,0,1,0,0,1,0,1],
    [1,0,0,1,1,0,1,1,1,0,1,1],
    [1,0,0,0,1,0,0,0,1,0,0,1],
    [1,0,0,1,1,1,1,0,0,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1]
])
# (10,7)は壁なので到達可能な通路に設定
goal_B = (9,7)
assert maze_B[goal_B[1], goal_B[0]] == 0, "goal_B は通路セル(0)にしてください"

# ===== スケジューラ =====
def expo_schedule(ep, total, start_val, end_val):
    if total <= 1: return end_val
    frac = ep / (total - 1)
    start_val = max(start_val, 1e-8)
    end_val = max(end_val, 1e-8)
    return start_val * (end_val / start_val) ** frac

def alpha_at(ep, total):
    return max(alpha_final, expo_schedule(ep, total, alpha_init, alpha_final))

def epsilon_at(ep, total, eps_start, eps_end=EPS_END):
    return max(eps_end, expo_schedule(ep, total, eps_start, eps_end))

# ===== 行動選択 =====
def choose_action_train(state, Q, epsilon):
    if random.random() < epsilon:
        return random.randint(0, len(actions)-1)
    return int(np.argmax(Q[state[1], state[0]]))  # Q[y,x,:]

def choose_action_greedy(state, Q):
    return int(np.argmax(Q[state[1], state[0]]))

# ===== Pygame（最小構成の描画だけ） =====
def init_screen(maze):
    pygame.init()
    W, H = maze.shape[1]*CELL, maze.shape[0]*CELL
    return pygame.display.set_mode((W, H))

def pump_events():
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

def draw_maze(screen, maze, agent_pos, goal):
    screen.fill(WHITE)
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            r = pygame.Rect(x*CELL, y*CELL, CELL, CELL)
            if maze[y,x] == 1:
                pygame.draw.rect(screen, BLACK, r)
            elif (x, y) == goal:
                pygame.draw.rect(screen, GREEN, r)
            pygame.draw.rect(screen, BLACK, r, 1)  # 枠線
    ax = agent_pos[0]*CELL + CELL//2
    ay = agent_pos[1]*CELL + CELL//2
    pygame.draw.circle(screen, BLUE, (ax, ay), CELL//4)
    pygame.display.flip()

# ===== Potential-based Reward Shaping =====
def phi(state, goal):
    return - (abs(state[0]-goal[0]) + abs(state[1]-goal[1]))

# ===== 学習のみ（訓練曲線用） =====
def run_q_learning(maze, goal, Q_init, episodes, eps_start=1.0, visualize=False, use_shaping=True):
    Q = np.copy(Q_init)  # Q[y, x, a]
    steps_list, rewards_list = [], []

    screen = init_screen(maze) if visualize else None

    for ep in range(episodes):
        epsilon = epsilon_at(ep, episodes, eps_start, EPS_END)
        alpha_t = alpha_at(ep, episodes)
        state = start
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < STEP_LIMIT:
            if screen:
                pump_events()
                draw_maze(screen, maze, state, goal)

            a = choose_action_train(state, Q, epsilon)
            dx, dy = actions[a]
            ns = (state[0] + dx, state[1] + dy)

            if maze[ns[1], ns[0]] == 1:
                reward = reward_wall
                ns = state
            elif ns == goal:
                reward = reward_goal
                done = True
            else:
                reward = reward_step

            shaped_r = reward + gamma * phi(ns, goal) - phi(state, goal) if use_shaping else reward
            target = shaped_r if done else shaped_r + gamma * np.max(Q[ns[1], ns[0]])
            Q[state[1], state[0], a] += alpha_t * (target - Q[state[1], state[0], a])

            state = ns
            total_reward += reward
            steps += 1

        steps_list.append(steps)
        rewards_list.append(total_reward)

    if screen:
        pygame.quit()
    return Q, steps_list, rewards_list

# ===== 評価ロールアウト（ε=0, α=0） =====
def rollout_greedy(Q, maze, goal, limit=STEP_LIMIT):
    state = start
    steps = 0
    total_reward = 0.0
    while state != goal and steps < limit:
        a = choose_action_greedy(state, Q)
        dx, dy = actions[a]
        ns = (state[0] + dx, state[1] + dy)

        if maze[ns[1], ns[0]] == 1:
            total_reward += reward_wall
            ns = state
        elif ns == goal:
            total_reward += reward_goal
        else:
            total_reward += reward_step

        state = ns
        steps += 1
    return steps, total_reward

# ===== 学習＋毎エピソード評価（評価曲線用） =====
def run_q_learning_with_eval(maze, goal, Q_init, episodes, eps_start=1.0,
                             visualize=False, use_shaping=True, eval_runs=5):
    Q = np.copy(Q_init)
    steps_list, rewards_list = [], []
    eval_steps, eval_rewards = [], []

    screen = init_screen(maze) if visualize else None

    for ep in range(episodes):
        epsilon = epsilon_at(ep, episodes, eps_start, EPS_END)
        alpha_t = alpha_at(ep, episodes)
        state = start
        total_reward = 0.0
        steps = 0
        done = False

        # --- 学習 ---
        while not done and steps < STEP_LIMIT:
            if screen:
                pump_events()
                draw_maze(screen, maze, state, goal)

            a = choose_action_train(state, Q, epsilon)
            dx, dy = actions[a]
            ns = (state[0] + dx, state[1] + dy)

            if maze[ns[1], ns[0]] == 1:
                reward = reward_wall
                ns = state
            elif ns == goal:
                reward = reward_goal
                done = True
            else:
                reward = reward_step

            shaped_r = reward + gamma * phi(ns, goal) - phi(state, goal) if use_shaping else reward
            target = shaped_r if done else shaped_r + gamma * np.max(Q[ns[1], ns[0]])
            Q[state[1], state[0], a] += alpha_t * (target - Q[state[1], state[0], a])

            state = ns
            total_reward += reward
            steps += 1

        steps_list.append(steps)
        rewards_list.append(total_reward)

        # --- 評価（学習オフ、ε=0, α=0） ---
        s_sum, r_sum = 0, 0.0
        for _ in range(eval_runs):
            s, r = rollout_greedy(Q, maze, goal, limit=STEP_LIMIT)
            s_sum += s; r_sum += r
        eval_steps.append(s_sum / eval_runs)
        eval_rewards.append(r_sum / eval_runs)

    if screen:
        pygame.quit()
    return Q, steps_list, rewards_list, eval_steps, eval_rewards

# ===== 転移ユーティリティ =====
def detect_newly_blocked(maze_A, maze_B):
    rows, cols = maze_B.shape
    newly = []
    for y in range(rows):
        for x in range(cols):
            if maze_B[y,x]==1 and maze_A[y,x]==0:
                newly.append((x,y))
    return newly

def mark_near(mask, pts, maze, radius=1):
    rows, cols = maze.shape
    for (x,y) in pts:
        for yy in range(max(0, y-radius), min(rows, y+radius+1)):
            for xx in range(max(0, x-radius), min(cols, x+radius+1)):
                if maze[yy,xx]==0:
                    mask[yy,xx] = True

def init_Q_with_obstacle_transfer(Q_A, maze_A, maze_B, beta=0.7, radius=1):
    Q_B = Q_A.copy()  # [y,x,a]
    rows, cols = maze_B.shape
    newly_blocked = detect_newly_blocked(maze_A, maze_B)

    # 壁に入る行動のQを0化
    for (x, y) in newly_blocked:
        for a, (dx, dy) in enumerate(actions):
            sx, sy = x - dx, y - dy
            if 0 <= sx < cols and 0 <= sy < rows and maze_B[sy, sx] == 0:
                Q_B[sy, sx, a] = 0.0

    # 壁セルのQも0化
    for (x, y) in newly_blocked:
        if 0 <= x < cols and 0 <= y < rows:
            Q_B[y, x, :] = 0.0

    # 近傍減衰
    def neighbors_within_r(xc, yc, r):
        for yy in range(max(0, yc - r), min(rows, yc + r + 1)):
            for xx in range(max(0, xc - r), min(cols, xc + r + 1)):
                if maze_B[yy, xx] == 0:
                    yield xx, yy

    for (x, y) in newly_blocked:
        for xx, yy in neighbors_within_r(x, y, radius):
            Q_B[yy, xx, :] *= beta

    return Q_B

def weaken_bias_toward_old_goal(Q_in, maze, goal_old, goal_new,
                                dir_beta=0.85, near_beta=0.7, radius=1,
                                zero_goals=True):
    Q = Q_in.copy()
    rows, cols = maze.shape
    def md(p, q): return abs(p[0]-q[0]) + abs(p[1]-q[1])

    for y in range(rows):
        for x in range(cols):
            if maze[y, x] == 1:
                continue
            d_old = md((x, y), goal_old)
            for a, (dx, dy) in enumerate(actions):
                nx, ny = x+dx, y+dy
                if 0 <= nx < cols and 0 <= ny < rows and maze[ny, nx] == 0:
                    if md((nx, ny), goal_old) < d_old:
                        Q[y, x, a] *= dir_beta

    gx, gy = goal_old
    for yy in range(max(0, gy - radius), min(rows, gy + radius + 1)):
        for xx in range(max(0, gx - radius), min(cols, x + radius + 1)):
            if maze[yy, xx] == 0:
                Q[yy, xx, :] *= near_beta

    if zero_goals:
        if 0 <= goal_old[0] < cols and 0 <= goal_old[1] < rows:
            Q[goal_old[1], goal_old[0], :] = 0.0
        if 0 <= goal_new[0] < cols and 0 <= goal_new[1] < rows:
            Q[goal_new[1], goal_new[0], :] = 0.0

    return Q

def promote_goal_entry_equally(Q_in, maze, goal_new, delta=20.0):
    Q = Q_in.copy()
    gx, gy = goal_new
    neighbors = [(gx, gy-1, (0, 1)), (gx+1, gy, (-1, 0)), (gx, gy+1, (0, -1)), (gx-1, gy, (1, 0))]
    rows, cols = maze.shape
    for px, py, (dx, dy) in neighbors:
        if 0 <= px < cols and 0 <= py < rows and maze[py, px] == 0:
            for a, (adx, ady) in enumerate(actions):
                if (adx, ady) == (dx, dy):
                    Q[py, px, a] += delta
                    break
    Q[goal_new[1], goal_new[0], :] = 0.0
    return Q

def lambda_scale_Q_from_A(Q_A, maze_A, maze_B, goal_old, goal_new,
                          newly_blocked_radius=1, goal_radius=1,
                          lambda_default=0.6, lambda_change=0.2):
    rows, cols = maze_B.shape
    mask_change = np.zeros((rows, cols), dtype=bool)
    newly = detect_newly_blocked(maze_A, maze_B)
    mark_near(mask_change, newly, maze_B, radius=newly_blocked_radius)
    mark_near(mask_change, [goal_old, goal_new], maze_B, radius=goal_radius)

    Q_scaled = np.zeros_like(Q_A)  # [y,x,a]
    for y in range(rows):
        for x in range(cols):
            lam = lambda_change if mask_change[y,x] else lambda_default
            Q_scaled[y, x, :] = lam * Q_A[y, x, :]
    return Q_scaled

# ===== 学習と転移（メイン） =====
random.seed(0); np.random.seed(0)

# 1) 転移元Aの学習（評価は不要）
Q_A_init = np.zeros((maze_A.shape[0], maze_A.shape[1], len(actions)))
Q_A, _, _ = run_q_learning(maze_A, goal_A, Q_A_init, episodes, eps_start=1.0, visualize=False, use_shaping=True)

# 2) B: No Transfer（学習＋評価）
Q_B_blank = np.zeros((maze_B.shape[0], maze_B.shape[1], len(actions)))
_, steps_nt, rewards_nt, eval_steps_nt, eval_rewards_nt = \
    run_q_learning_with_eval(maze_B, goal_B, Q_B_blank, episodes, eps_start=1.0, visualize=False, use_shaping=True, eval_runs=5)

# 3) B: Transfer (Copy only)（学習＋評価）
Q_B_init_copy = np.copy(Q_A)
_, steps_cp, rewards_cp, eval_steps_cp, eval_rewards_cp = \
    run_q_learning_with_eval(maze_B, goal_B, Q_B_init_copy, episodes, eps_start=0.3, visualize=False, use_shaping=True, eval_runs=5)

# 4) B: Transfer (λ+Obstacle+OldGoalWeak+GoalEntryBoost)（学習＋評価）
Q_B_init = lambda_scale_Q_from_A(Q_A, maze_A, maze_B, goal_A, goal_B,
                                 newly_blocked_radius=1, goal_radius=1,
                                 lambda_default=0.6, lambda_change=0.2)
Q_B_init = init_Q_with_obstacle_transfer(Q_B_init, maze_A, maze_B, beta=0.7, radius=1)
Q_B_init = weaken_bias_toward_old_goal(Q_B_init, maze_B, goal_old=goal_A, goal_new=goal_B,
                                       dir_beta=0.85, near_beta=0.7, radius=1, zero_goals=True)
Q_B_init = promote_goal_entry_equally(Q_B_init, maze_B, goal_new=goal_B, delta=20.0)
_, steps_ad, rewards_ad, eval_steps_ad, eval_rewards_ad = \
    run_q_learning_with_eval(maze_B, goal_B, Q_B_init, episodes, eps_start=0.3, visualize=False, use_shaping=True, eval_runs=5)

# ===== 図1: 訓練カーブ（ε>0, α>0 の学習ログ） =====
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(steps_nt, label='B: No Transfer')
plt.plot(steps_cp, label='B: Transfer (Copy only)')
plt.plot(steps_ad, label='B: Transfer (λ+Obstacle+OldGoalWeak+GoalEntryBoost)')
plt.title('Training: Steps per Episode'); plt.xlabel('Episode'); plt.ylabel('Steps'); plt.legend()

plt.subplot(1,2,2)
plt.plot(rewards_nt, label='B: No Transfer')
plt.plot(rewards_cp, label='B: Transfer (Copy only)')
plt.plot(rewards_ad, label='B: Transfer (λ+Obstacle+OldGoalWeak+GoalEntryBoost)')
plt.title('Training: Total Reward per Episode'); plt.xlabel('Episode'); plt.ylabel('Reward'); plt.legend()
plt.tight_layout(); plt.show()

# ===== 図2: 評価カーブ（ε=0, α=0 の平均） =====
x = np.arange(episodes)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(x, eval_steps_nt, label='Eval: No Transfer')
plt.plot(x, eval_steps_cp, label='Eval: Transfer (Copy only)')
plt.plot(x, eval_steps_ad, label='Eval: Transfer (λ+Obstacle+OldGoalWeak+GoalEntryBoost)')
plt.title('Evaluation (ε=0, α=0): Steps'); plt.xlabel('Episode'); plt.ylabel('Steps'); plt.legend()

plt.subplot(1,2,2)
plt.plot(x, eval_rewards_nt, label='Eval: No Transfer')
plt.plot(x, eval_rewards_cp, label='Eval: Transfer (Copy only)')
plt.plot(x, eval_rewards_ad, label='Eval: Transfer (λ+Obstacle+OldGoalWeak+GoalEntryBoost)')
plt.title('Evaluation (ε=0, α=0): Total Reward'); plt.xlabel('Episode'); plt.ylabel('Reward'); plt.legend()
plt.tight_layout(); plt.show()
