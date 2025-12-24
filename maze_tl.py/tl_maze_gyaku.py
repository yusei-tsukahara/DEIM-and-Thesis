import numpy as np
import random
import matplotlib.pyplot as plt
import pygame
import time

# 行動・学習パラメータ
actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # 左・下・右・上
alpha = 0.5
gamma = 0.99
epsilon_init = 1.0
epsilon_decay = 0.90
epsilon_min = 0.01
episodes = 150

# 報酬
reward_goal = 100.0
reward_wall = -10.0
reward_step = -1.0

# 描画設定
CELL = 30
WHITE = (255,255,255)
BLACK = (0,0,0)
BLUE = (0,0,255)
GREEN = (0,200,0)

# Maze A
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

# Maze B
maze_B = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,0,0,0,0,0,0,1],
    [1,0,1,0,1,0,1,1,1,1,0,1],
    [1,0,1,0,0,0,0,0,1,0,0,1],
    [1,0,1,1,1,1,1,0,1,0,1,1],
    [1,0,0,0,0,0,1,0,0,0,0,1],
    [1,1,1,1,1,0,1,1,1,1,0,1],
    [1,0,0,0,1,0,0,0,0,1,0,1],
    [1,0,1,0,1,1,1,1,0,1,0,1],
    [1,0,1,0,0,0,0,1,0,0,0,1],
    [1,0,0,0,1,1,0,0,0,1,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1]
])

# ボルツマン選択
def choose_action(state, Q, epsilon):
    tau = max(1.0, epsilon)
    q_values = Q[state[0], state[1]]
    q_values = np.clip(q_values, -500, 500)
    exp_q = np.exp((q_values - np.max(q_values)) / tau)
    probs = exp_q / np.sum(exp_q)
    return np.random.choice(len(actions), p=probs)

# 迷路描画
def draw_maze(screen, maze, agent_pos, goal):
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

# Q学習
def run_q_learning(maze, Q_init, episodes, start, goal, visualize=False):
    Q = np.copy(Q_init)
    steps_list = []
    rewards_list = []
    epsilon = epsilon_init

    if visualize:
        pygame.init()
        WIDTH, HEIGHT = maze.shape[1]*CELL, maze.shape[0]*CELL
        screen = pygame.display.set_mode((WIDTH, HEIGHT))

    for ep in range(episodes):
        state = start
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 500:
            if visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
                draw_maze(screen, maze, state, goal)
                time.sleep(0.0)

            action_idx = choose_action(state, Q, epsilon)
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

        steps_list.append(steps)
        rewards_list.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if visualize:
        pygame.quit()

    return Q, steps_list, rewards_list

# スタートとゴールを迷路ごとに設定
start_A = (10, 10)
goal_A = (1, 1)

start_B = (1, 1)
goal_B = (10, 10)

# Maze Aで学習
Q_A_init = np.zeros((maze_A.shape[0], maze_A.shape[1], len(actions)))
Q_A, _, _ = run_q_learning(maze_A, Q_A_init, episodes, start_A, goal_A, visualize=False)

# Maze Bを転移なしで学習
Q_B_blank = np.zeros((maze_B.shape[0], maze_B.shape[1], len(actions)))
_, steps_B_no_transfer, rewards_B_no_transfer = run_q_learning(maze_B, Q_B_blank, episodes, start_B, goal_B, visualize=True)

# Maze Bを転移ありで学習（Q_Aを初期値として使う）
Q_B_transfer, steps_B_transfer, rewards_B_transfer = run_q_learning(maze_B, Q_A, episodes, start_B, goal_B, visualize=True)

# 学習曲線表示
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.plot(steps_B_no_transfer, label='Maze B (No Transfer)')
plt.plot(steps_B_transfer, label='Maze B (Transfer)')
plt.title("Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.legend()

plt.subplot(1,2,2)
plt.plot(rewards_B_no_transfer, label='Maze B (No Transfer)')
plt.plot(rewards_B_transfer, label='Maze B (Transfer)')
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()

plt.tight_layout()
plt.show()
