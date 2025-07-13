# 파일명: dqn_app_ultimate.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- 훈련 코드와 동일한 환경 및 DQN 모델 정의 ---
actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
SIZE = 10 # 훈련된 모델과 동일한 크기
MAX_STEPS = 500 # GUI에서는 넉넉한 스텝 제공
ANIMATION_DELAY = 50 # 애니메이션 속도 (ms)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_shortest_path_bfs(env, start, goal):
    size = env.shape[0]; queue = deque([start]); visited = {start}; parent = {start: None}
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            path = []; curr = goal
            while curr is not None: path.append(curr); curr = parent.get(curr)
            return path[::-1]
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and env[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc)); queue.append((nr, nc)); parent[(nr, nc)] = (r, c)
    return []

def generate_map_with_guaranteed_path(size, obstacle_prob=0.25):
    print("경로가 보장된 맵을 생성하는 중...")
    while True:
        env = np.random.choice([0, 1], size=(size, size), p=[1 - obstacle_prob, obstacle_prob])
        start = tuple(np.random.randint(0, size, size=2)); goal = tuple(np.random.randint(0, size, size=2))
        if env[start] == 1 or env[goal] == 1 or start == goal: continue
        env[start] = 0; env[goal] = 0
        if find_shortest_path_bfs(env, start, goal):
            print("성공: 풀 수 있는 맵을 생성했습니다.")
            return env, start, goal

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__(); self.conv1 = nn.Conv2d(5, 16, kernel_size=3, padding=1); self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1); self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1); self.bn3 = nn.BatchNorm2d(64)
        self.head = nn.Linear(w * h * 64, outputs)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

def get_state_tensor(env, agent_pos, goal_pos):
    size = env.shape[0]
    obstacle_map = torch.from_numpy(env).float().unsqueeze(0); agent_map = torch.zeros_like(obstacle_map)
    if agent_pos: agent_map[0, agent_pos[0], agent_pos[1]] = 1
    goal_map = torch.zeros_like(obstacle_map)
    if goal_pos: goal_map[0, goal_pos[0], goal_pos[1]] = 1
    dx = (goal_pos[1] - agent_pos[1]) / (size - 1); dy = (goal_pos[0] - agent_pos[0]) / (size - 1)
    dx_map = torch.full_like(obstacle_map, dx); dy_map = torch.full_like(obstacle_map, dy)
    return torch.cat([obstacle_map, agent_map, goal_map, dx_map, dy_map], dim=0).unsqueeze(0).to(device)

# --- GUI 애플리케이션 클래스 ---
class PathfindingGUI:
    def __init__(self, master):
        self.master = master
        master.title("DQN Pathfinding Simulator - Ultimate")
        self.agent_model = self.load_model()
        if self.agent_model is None: master.destroy(); return
        self.fig = plt.Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        button_frame = tk.Frame(master); button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        self.btn_find_path = tk.Button(button_frame, text="경로 찾기", command=self.start_pathfinding_animation)
        self.btn_find_path.pack(side=tk.LEFT, expand=True, padx=5)
        self.btn_new_map = tk.Button(button_frame, text="새로운 맵 생성", command=self.new_map)
        self.btn_new_map.pack(side=tk.LEFT, expand=True, padx=5)
        self.btn_quit = tk.Button(button_frame, text="종료", command=master.quit)
        self.btn_quit.pack(side=tk.LEFT, expand=True, padx=5)
        self.new_map()

    def load_model(self):
        MODEL_PATH = "dqn\\dqn_model_ultimate.pt"
        try:
            model = DQN(SIZE, SIZE, len(actions)).to(device)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            print(f"성공: '{MODEL_PATH}' 모델을 불러왔습니다.")
            return model
        except FileNotFoundError: messagebox.showerror("오류", f"'{MODEL_PATH}' 파일을 찾을 수 없습니다."); return None
        except Exception as e: messagebox.showerror("오류", f"모델 로딩 오류: {e}"); return None

    def new_map(self):
        self.env, self.start, self.goal = generate_map_with_guaranteed_path(size=SIZE)
        self.draw_map()

    def draw_map(self, agent_path=None):
        self.ax.clear(); visual = np.stack([self.env.copy().astype(float)] * 3, axis=-1)
        if agent_path:
            for pos in agent_path[:-1]:
                if pos != self.start and pos != self.goal: visual[pos] = [1, 1, 0] # Yellow
            current_pos = agent_path[-1]
            if current_pos != self.start and current_pos != self.goal:
                visual[current_pos] = [1, 0, 0] # Red
        visual[self.start] = [0, 1, 0]; visual[self.goal] = [0, 0, 1]
        self.ax.imshow(visual); self.ax.set_title("DQN Pathfinding Simulator"); self.ax.set_xticks([]); self.ax.set_yticks([])
        self.canvas.draw()
        
    def start_pathfinding_animation(self):
        print("훈련된 모델로 경로 탐색을 시작합니다..."); self.btn_find_path.config(state=tk.DISABLED); self.btn_new_map.config(state=tk.DISABLED)
        self.state = self.start; self.agent_path = [self.state]; self.step_count = 0
        self.master.after(ANIMATION_DELAY, self._step)

    def _step(self):
        if self.state == self.goal or self.step_count >= MAX_STEPS:
            if self.state == self.goal: print("탐색 성공: 목표에 도달했습니다!")
            else: print("탐색 실패: 최대 스텝 내에 목표에 도달하지 못했습니다.")
            self.btn_find_path.config(state=tk.NORMAL); self.btn_new_map.config(state=tk.NORMAL)
            return
        
        state_tensor = get_state_tensor(self.env, self.state, self.goal)
        with torch.no_grad():
            action_idx = self.agent_model(state_tensor).max(1)[1].item()
        
        dx, dy = actions[action_idx]
        next_state_candidate = (self.state[0] + dx, self.state[1] + dy)
        
        # 벽에 부딪히면 제자리에 머무는 로직
        if (0 <= next_state_candidate[0] < SIZE and 
            0 <= next_state_candidate[1] < SIZE and 
            self.env[next_state_candidate] == 0):
            self.state = next_state_candidate
        
        self.agent_path.append(self.state)
        self.step_count += 1
        self.draw_map(agent_path=self.agent_path)
        self.master.after(ANIMATION_DELAY, self._step)

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PathfindingGUI(root)
    root.mainloop()