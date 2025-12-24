import numpy as np
import random
import pygame
import matplotlib.pyplot as plt
import time

# ---------- 迷路 ----------
# ---------- 迷路（6x6） ----------
# ---------- 迷路（7x7） ----------
maze = np.array([
    [1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1],
    [1,0,1,1,1,0,1],
    [1,0,0,0,1,0,1],
    [1,1,0,1,0,0,1],
    [1,0,0,0,0,0,1],
    [1,1,1,1,1,1,1]
])

start = (1, 1)   # 左上の空白
goal  = (5, 5)   # 右下の空白


actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # 上右下左

# ---------- パラメータ ----------
Q = np.zeros((maze.shape[0], maze.shape[1], len(actions)))
alpha = 0.5
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.90
epsilon_min = 0.01
episodes = 50

reward_goal = 100.0
reward_wall = -10.0
reward_step = -1.0

selection_mode = "boltzmann"  # ← "epsilon_greedy" か "boltzmann" を切り替える

# ---------- PyGame ----------
CELL = 30
pygame.init()
WIDTH, HEIGHT = maze.shape[1]*CELL, maze.shape[0]*CELL
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Q-Learning Maze")

WHITE = (255,255,255)
BLACK = (0,0,0)
BLUE = (0,0,255)
GREEN = (0,200,0)

def draw_maze(agent_pos):
    screen.fill(WHITE)
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            rect = pygame.Rect(x*CELL, y*CELL, CELL, CELL)
            if maze[y,x] == 1:
                pygame.draw.rect(screen, BLACK, rect)
            elif (x, y) == goal:
                pygame.draw.rect(screen, GREEN, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)
    pygame.draw.circle(screen, BLUE, (agent_pos[0]*CELL + CELL//2, agent_pos[1]*CELL + CELL//2), CELL//4)
    pygame.display.flip()

# ---------- 行動選択 ----------
def choose_action(state, Q, epsilon, mode):
    if mode == "epsilon_greedy":
        if random.random() < epsilon:
            return random.randint(0, len(actions)-1)
        else:
            return np.argmax(Q[state[0], state[1]])
    elif mode == "boltzmann":
        tau = max(1.0, epsilon)  # 最低1.0で安定
        q_values = Q[state[0], state[1]]
        q_values = np.clip(q_values, -500, 500)  # Q値制限
        exp_q = np.exp((q_values - np.max(q_values)) / tau)  # オーバーフロー防止
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(len(actions), p=probs)
    else:
        raise ValueError("Unknown selection mode")

# ---------- 学習 ----------
steps_per_episode = []
rewards_per_episode = []

for ep in range(episodes):
    state = start
    total_reward = 0
    steps = 0
    done = False

    while not done and steps < 500:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        draw_maze(state)
        time.sleep(0.0)

        action_idx = choose_action(state, Q, epsilon, selection_mode)
        move = actions[action_idx]
        next_state = (state[0] + move[0], state[1] + move[1])

        if maze[next_state[1], next_state[0]] == 1:
            reward = reward_wall
            next_state = state
        elif next_state == goal:
            reward = reward_goal
            done = True
        else:
            reward = reward_step

        best_next_Q = np.max(Q[next_state[0], next_state[1]])
        Q[state[0], state[1], action_idx] += alpha * (reward + gamma * best_next_Q - Q[state[0], state[1], action_idx])

        state = next_state
        total_reward += reward
        steps += 1

    steps_per_episode.append(steps)
    rewards_per_episode.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (ep+1) % 50 == 0 or ep == 0:
        print(f"Episode {ep+1}: Steps={steps}, Reward={total_reward:.2f}, Epsilon/Tau={epsilon:.3f}")

# ---------- グラフ（表示のみ・保存なし） ----------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(steps_per_episode)
plt.title("Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")

plt.subplot(1,2,2)
plt.plot(rewards_per_episode)
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")

plt.tight_layout()
plt.show()

# ===== Qテーブル可視化 =====
def visualize_Q(maze, Q, start, goal, actions):
    import numpy as np
    import matplotlib.pyplot as plt

    # Q は学習側で [x, y] 参照になっているので、可視化では転置して整合
    Qxy = np.transpose(Q, (1, 0, 2))  # 以降は [y, x, a] っぽく扱う

    rows, cols = maze.shape
    maxQ = np.full((rows, cols), np.nan)
    U = np.zeros((rows, cols))  # 矢印 x 成分
    V = np.zeros((rows, cols))  # 矢印 y 成分（origin='upper' に合わせて符号はそのまま）

    for y in range(rows):
        for x in range(cols):
            if maze[y, x] == 1:
                continue
            vals = Qxy[x, y]
            a = int(np.argmax(vals))
            maxQ[y, x] = float(np.max(vals))
            dx, dy = actions[a]
            U[y, x] = dx
            V[y, x] = dy  # ★ ここを反転しない（以前は -dy だった）

    fig = plt.figure(figsize=(12, 5))

    # 1) 方策の矢印
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(maze, cmap='gray_r', origin='upper')  # ★ pygame と同じ向き
    ys, xs = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

    mask = (maze == 1)
    U_masked = np.ma.array(U, mask=mask)
    V_masked = np.ma.array(V, mask=mask)

    ax1.quiver(xs, ys, U_masked, V_masked, pivot='middle', scale=3, width=0.01)
    ax1.scatter([start[0], goal[0]], [start[1], goal[1]], s=100, marker='o')
    ax1.text(start[0], start[1], 'S', ha='center', va='center', fontsize=12, weight='bold')
    ax1.text(goal[0],  goal[1],  'G', ha='center', va='center', fontsize=12, weight='bold')
    ax1.set_title('Greedy policy (argmax Q)')
    ax1.set_xticks(np.arange(cols)); ax1.set_yticks(np.arange(rows))
    ax1.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)
    # ★ 向きを固定（上が y=0）
    ax1.set_xlim(-0.5, cols - 0.5)
    ax1.set_ylim(rows - 0.5, -0.5)

    # 2) 各マスの最大Q値
    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(maxQ, origin='upper')  # ★ 同じく上が0
    ax2.text(start[0], start[1], 'S', ha='center', va='center', fontsize=12, weight='bold')
    ax2.text(goal[0],  goal[1],  'G', ha='center', va='center', fontsize=12, weight='bold')
    ax2.set_title('Max Q per cell')
    ax2.set_xticks(np.arange(cols)); ax2.set_yticks(np.arange(rows))
    ax2.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
    
visualize_Q(maze, Q, start, goal, actions)
