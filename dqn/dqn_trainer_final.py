# 파일명: check1.py (또는 dqn_app_final.py)

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
SIZE = 15
MAX_STEPS = 1000
ANIMATION_DELAY = 10
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

def generate_map_with_guaranteed_path(size=15, obstacle_prob=0.3):
    print("경로가 보장된 맵을 생성하는 중...")
    while True:
        env = np.random.choice([0, 1], size=(size, size), p=[1-obstacle_prob, obstacle_prob])
        start = tuple(np.random.randint(0, size, size=2)); goal = tuple(np.random.randint(0, size, size=2))
        if env[start] == 1 or env[goal] == 1 or start == goal: continue
        env[start] = 0; env[goal] = 0
        if find_shortest_path_bfs(env, start, goal):
            print("성공: 풀 수 있는 맵을 생성했습니다.")
            return env, start, goal

# --- 훈련된 모델과 동일한 '3채널' 입력 구조 ---
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.head = nn.Linear(w * h * 64, outputs)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

def get_state_tensor(env, agent_pos, goal_pos):
    obstacle_map = torch.from_numpy(env).float().unsqueeze(0)
    agent_map = torch.zeros_like(obstacle_map)
    if agent_pos: agent_map[0, agent_pos[0], agent_pos[1]] = 1
    goal_map = torch.zeros_like(obstacle_map)
    if goal_pos: goal_map[0, goal_pos[0], goal_pos[1]] = 1
    return torch.cat([obstacle_map, agent_map, goal_map], dim=0).unsqueeze(0).to(device)

# --- GUI 애플리케이션 클래스 ---
class PathfindingGUI:
    def __init__(self, master):
        self.master = master
        master.title("DQN Pathfinding Simulator (Hybrid Ver.)")
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
        MODEL_PATH = "dqn_model_generalized.pt"
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
                if pos != self.start and pos != self.goal: visual[pos] = [1, 1, 0]
            current_pos = agent_path[-1]
            if current_pos != self.start and current_pos != self.goal: visual[current_pos] = [1, 0, 0]
        visual[self.start] = [0, 1, 0]; visual[self.goal] = [0, 0, 1]
        self.ax.imshow(visual); self.ax.set_title("DQN Pathfinding Simulator"); self.ax.set_xticks([]); self.ax.set_yticks([])
        self.canvas.draw()
        
    def start_pathfinding_animation(self):
        print("하이브리드 모델로 경로 탐색을 시작합니다..."); self.btn_find_path.config(state=tk.DISABLED); self.btn_new_map.config(state=tk.DISABLED)
        self.state = self.start
        # ★★★★★ 핵심 수정 1: '단기 기억'을 위한 경로 기록용 deque 추가 ★★★★★
        self.agent_path = deque([self.state], maxlen=10) # 최근 10개의 경로만 기억
        self.step_count = 0
        self.master.after(ANIMATION_DELAY, self._step)

    def _step(self):
        if self.state == self.goal or self.step_count >= MAX_STEPS:
            if self.state == self.goal: print("탐색 성공: 목표에 도달했습니다!")
            else: print("탐색 실패: 최대 스텝 내에 목표에 도달하지 못했습니다.")
            self.btn_find_path.config(state=tk.NORMAL); self.btn_new_map.config(state=tk.NORMAL)
            return

        # Plan A: DQN 모델에게 행동을 물어봄
        state_tensor = get_state_tensor(self.env, self.state, self.goal)
        with torch.no_grad(): action_idx = self.agent_model(state_tensor).max(1)[1].item()
        dx, dy = actions[action_idx]
        next_state_candidate = (self.state[0] + dx, self.state[1] + dy)
        
        # ★★★★★ 핵심 수정 2: '루프 감지' 및 '강제 탈출' 로직 추가 ★★★★★
        is_stuck_in_loop = next_state_candidate in self.agent_path
        is_wall_hit = not (0 <= next_state_candidate[0] < SIZE and 0 <= next_state_candidate[1] < SIZE and self.env[next_state_candidate] == 0)

        if is_wall_hit or is_stuck_in_loop:
            if is_stuck_in_loop: print(f"루프 감지! 현재 위치({self.state})에서 강제 탈출 실행...")
            else: print(f"DQN 판단 막힘! 현재 위치({self.state})에서 휴리스틱 탐색 실행...")
            
            # Plan B: 휴리스틱으로 최선의 길 찾기
            best_next_state = None
            min_dist = float('inf')

            # 현재 위치에서 상/하/좌/우를 모두 시도
            shuffled_actions = list(actions.values())
            random.shuffle(shuffled_actions) # 매번 다른 순서로 탐색하여 다양성 확보
            
            for move in shuffled_actions:
                temp_next = (self.state[0] + move[0], self.state[1] + move[1])
                
                # 이동 가능하고, 아직 가보지 않은 새로운 길 중에서
                if (0 <= temp_next[0] < SIZE and 0 <= temp_next[1] < SIZE and self.env[temp_next] == 0):
                    # 목표와 가장 가까워지는 길을 찾는다
                    dist = np.sqrt((temp_next[0] - self.goal[0])**2 + (temp_next[1] - self.goal[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_next_state = temp_next
            
            # 만약 모든 길이 막혀있거나 이미 가본 길이라면, 어쩔 수 없이 제자리에 머묾
            if best_next_state is None:
                self.state = self.state
            else:
                self.state = best_next_state

        else:
            # DQN의 판단이 유효하면 그대로 따름
            self.state = next_state_candidate
        
        self.agent_path.append(self.state)
        self.step_count += 1
        self.draw_map(agent_path=list(self.agent_path)) # 그리기 위해 list로 변환
        self.master.after(ANIMATION_DELAY, self._step)

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PathfindingGUI(root)
    root.mainloop()