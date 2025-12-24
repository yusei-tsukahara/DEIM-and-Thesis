import numpy as np
import pygame
import time
import matplotlib.pyplot as plt

# ---------- 迷路 ----------
maze = np.array([
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

start = (1, 1)
goal = (10, 10)
actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # 上右下左

# ---------- パラメータ ----------
gamma = 0.99
theta = 1e-4  # 収束判定
reward_goal = 100.0
reward_wall = -10.0
reward_step = -1.0

# ---------- PyGame ----------
CELL = 30
pygame.init()
WIDTH, HEIGHT = maze.shape[1]*CELL, maze.shape[0]*CELL
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Value Iteration Maze")

WHITE = (255,255,255)
BLACK = (0,0,0)
BLUE = (0,0,255)
GREEN = (0,200,0)
RED = (255,0,0)

def draw_maze(agent_pos=None, path=[]):
    screen.fill(WHITE)
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            rect = pygame.Rect(x*CELL, y*CELL, CELL, CELL)
            if maze[y,x] == 1:
                pygame.draw.rect(screen, BLACK, rect)
            elif (x, y) == goal:
                pygame.draw.rect(screen, GREEN, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)
    for (px, py) in path:
        pygame.draw.circle(screen, RED, (px*CELL + CELL//2, py*CELL + CELL//2), CELL//5)
    if agent_pos:
        pygame.draw.circle(screen, BLUE, (agent_pos[0]*CELL + CELL//2, agent_pos[1]*CELL + CELL//2), CELL//4)
    pygame.display.flip()

# ---------- 報酬関数 ----------
def get_reward(state):
    if state == goal:
        return reward_goal
    return reward_step

# ---------- 遷移可能か判定 ----------
def is_valid(pos):
    x, y = pos
    if x < 0 or x >= maze.shape[1] or y < 0 or y >= maze.shape[0]:
        return False
    return maze[y, x] == 0 or (x, y) == goal

# ---------- 価値反復 ----------
V = np.zeros((maze.shape[0], maze.shape[1]))  # 状態価値
policy = np.zeros((maze.shape[0], maze.shape[1]), dtype=int)  # 最適方策（行動インデックス）

iteration = 0
while True:
    delta = 0
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            if maze[y, x] == 1 or (x, y) == goal:
                continue  # 壁やゴールは更新しない
            v = V[y, x]
            values = []
            for a_idx, (dx, dy) in enumerate(actions):
                nx, ny = x + dx, y + dy
                if is_valid((nx, ny)):
                    r = get_reward((nx, ny))
                    values.append(r + gamma * V[ny, nx])
                else:
                    values.append(reward_wall + gamma * V[y, x])  # 壁に当たる場合
            V[y, x] = max(values)
            policy[y, x] = np.argmax(values)
            delta = max(delta, abs(v - V[y, x]))
    iteration += 1
    if delta < theta:
        break

print(f"価値反復の収束: {iteration} 回の更新")

# ---------- 最短経路を追跡 ----------
path = []
pos = start
while pos != goal:
    path.append(pos)
    a_idx = policy[pos[1], pos[0]]
    dx, dy = actions[a_idx]
    next_pos = (pos[0] + dx, pos[1] + dy)
    if not is_valid(next_pos) or next_pos == pos:
        break
    pos = next_pos
path.append(goal)

# ---------- 結果の可視化 ----------
for p in path:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    draw_maze(agent_pos=p, path=path)
    time.sleep(0.2)

pygame.quit()

# ---------- 価値関数のヒートマップ ----------
plt.imshow(V, cmap='coolwarm')
plt.colorbar()
plt.title("State Value Function (V)")
plt.show()
