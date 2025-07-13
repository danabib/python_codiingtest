# 파일명: dqn_app_ultimate_back.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
SIZE = 10
MAX_STEPS = 500
ANIMATION_DELAY = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_shortest_path_bfs(env, start, goal):
    size = env.shape[0]
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            path = []
            curr = goal
            while curr is not None:
                path.append(curr)
                curr = parent.get(curr)
            return path[::-1]
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and env[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
                parent[(nr, nc)] = (r, c)
    return []

def generate_map_with_guaranteed_path(size, obstacle_prob=0.25):
    print("경로가 보장된 맵을 생성하는 중...")
    while True:
        env = np.random.choice([0, 1], size=(size, size), p=[1 - obstacle_prob, obstacle_prob])
        start = tuple(np.random.randint(0, size, size=2))
        goal = tuple(np.random.randint(0, size, size=2))
        if env[start] == 1 or env[goal] == 1 or start == goal:
            continue
        env[start] = 0
        env[goal] = 0
        if find_shortest_path_bfs(env, start, goal):
            print("성공: 풀 수 있는 맵을 생성했습니다.")
            return env, start, goal

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(5, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.head = nn.Linear(w * h * 64, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

def get_state_tensor(env, agent_pos, goal_pos):
    size = env.shape[0]
    obstacle_map = torch.from_numpy(env).float().unsqueeze(0)
    agent_map = torch.zeros_like(obstacle_map)
    if agent_pos:
        agent_map[0, agent_pos[0], agent_pos[1]] = 1
    goal_map = torch.zeros_like(obstacle_map)
    if goal_pos:
        goal_map[0, goal_pos[0], goal_pos[1]] = 1
    dx = (goal_pos[1] - agent_pos[1]) / (size - 1)
    dy = (goal_pos[0] - agent_pos[0]) / (size - 1)
    dx_map = torch.full_like(obstacle_map, dx)
    dy_map = torch.full_like(obstacle_map, dy)
    return torch.cat([obstacle_map, agent_map, goal_map, dx_map, dy_map], dim=0).unsqueeze(0).to(device)

class PathfindingGUI:
    def __init__(self, master):
        self.master = master
        master.title("DQN Pathfinding Simulator - Ultimate")
        self.agent_model = self.load_model()
        if self.agent_model is None:
            master.destroy()
            return
        self.fig = plt.Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        button_frame = tk.Frame(master)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        self.btn_find_path = tk.Button(button_frame, text="경로 찾기", command=self.start_pathfinding_animation)
        self.btn_find_path.pack(side=tk.LEFT, expand=True, padx=5)
        self.btn_new_map = tk.Button(button_frame, text="새로운 맵 생성", command=self.new_map)
        self.btn_new_map.pack(side=tk.LEFT, expand=True, padx=5)
        self.btn_quit = tk.Button(button_frame, text="종료", command=master.quit)
        self.btn_quit.pack(side=tk.LEFT, expand=True, padx=5)
        self.visited_counter = defaultdict(int)
        self.new_map()

    def load_model(self):
        MODEL_PATH = "dqn/dqn_model_ultimate.pt"
        try:
            model = DQN(SIZE, SIZE, len(actions)).to(device)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            print(f"성공: '{MODEL_PATH}' 모델을 불러왔습니다.")
            return model
        except FileNotFoundError:
            messagebox.showerror("오류", f"'{MODEL_PATH}' 파일을 찾을 수 없습니다.")
            return None
        except Exception as e:
            messagebox.showerror("오류", f"모델 로딩 오류: {e}")
            return None

    def new_map(self):
        self.env, self.start, self.goal = generate_map_with_guaranteed_path(size=SIZE)
        self.draw_map()

    def draw_map(self, agent_path=None):
        self.ax.clear()
        visual = np.stack([self.env.copy().astype(float)] * 3, axis=-1)
        if agent_path:
            for pos in agent_path[:-1]:
                if pos != self.start and pos != self.goal:
                    visual[pos] = [1, 1, 0]
            current_pos = agent_path[-1]
            if current_pos != self.start and current_pos != self.goal:
                visual[current_pos] = [1, 0, 0]
        visual[self.start] = [0, 1, 0]
        visual[self.goal] = [0, 0, 1]
        self.ax.imshow(visual)
        self.ax.set_title("DQN Pathfinding Simulator")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

    def start_pathfinding_animation(self):
        self.state = self.start
        self.prev_state = None
        self.backtracking = False
        self.agent_path = [self.state]
        self.visited_counter = defaultdict(int)
        self.step_count = 0
        self.btn_find_path.config(state=tk.DISABLED)
        self.btn_new_map.config(state=tk.DISABLED)
        self.master.after(ANIMATION_DELAY, self._step)

    def _step(self):
        # 종료 조건
        if self.state == self.goal or self.step_count >= MAX_STEPS:
            print("탐색 완료: ", "성공" if self.state == self.goal else "실패")
            self.btn_find_path.config(state=tk.NORMAL)
            self.btn_new_map.config(state=tk.NORMAL)
            return

        old_state = self.state
        self.visited_counter[self.state] += 1

        # 모델 예측 및 Q값 정렬
        state_tensor = get_state_tensor(self.env, self.state, self.goal)
        with torch.no_grad():
            q_values = self.agent_model(state_tensor).cpu().numpy()[0]
        sorted_actions = np.argsort(q_values)[::-1]

        # 목적지 방향 우선순위 조정
        curr_dist = abs(self.state[0] - self.goal[0]) + abs(self.state[1] - self.goal[1])
        toward, away = [], []
        for action_idx in sorted_actions:
            dx, dy = actions[action_idx]; ns = (self.state[0]+dx, self.state[1]+dy)
            if 0 <= ns[0] < SIZE and 0 <= ns[1] < SIZE and self.env[ns] == 0:
                new_dist = abs(ns[0]-self.goal[0]) + abs(ns[1]-self.goal[1])
                (toward if new_dist<curr_dist else away).append(action_idx)
        biased_actions = toward + away

        # 반복 방문 처리
        if self.visited_counter[self.state]>=2 and not self.backtracking:
            for act in biased_actions:
                dx, dy = actions[act]; ns = (self.state[0]+dx, self.state[1]+dy)
                if 0<=ns[0]<SIZE and 0<=ns[1]<SIZE and self.env[ns]==0 and self.visited_counter[ns]==0:
                    self.state=ns; break
            else:
                # 탈출 불가 -> 종료
                print(f"반복 위치 {self.state}에서 더 이상 이동 불가, 탐색 종료")
                self.btn_find_path.config(state=tk.NORMAL)
                self.btn_new_map.config(state=tk.NORMAL)
                return

        # 기본 이동
        self.backtracking=False; self.prev_state=old_state
        for act in biased_actions:
            dx, dy = actions[act]; ns=(old_state[0]+dx, old_state[1]+dy)
            if 0<=ns[0]<SIZE and 0<=ns[1]<SIZE and self.env[ns]==0:
                self.state=ns; break

        # 멈춤 감지: 위치 변화 없으면 종료
        if self.state==old_state:
            print(f"더 이상의 이동 경로 없음, 탐색 종료 at {self.state}")
            self.btn_find_path.config(state=tk.NORMAL)
            self.btn_new_map.config(state=tk.NORMAL)
            return

        # 시각화 및 다음 스텝
        self.agent_path.append(self.state)
        self.step_count+=1
        self.draw_map(agent_path=self.agent_path)
        self.master.after(ANIMATION_DELAY, self._step)

if __name__ == "__main__":
    root=tk.Tk(); app=PathfindingGUI(root); root.mainloop()
