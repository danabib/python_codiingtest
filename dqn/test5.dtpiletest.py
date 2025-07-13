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

# ------------------------------------------------------------------
# 1. 환경 및 DQN 모델 정의 (이전 코드와 거의 동일)
# ------------------------------------------------------------------

# BFS 및 맵 생성 함수
def find_shortest_path_bfs(env, start, goal):
    size = env.shape[0]
    queue = deque([start])
    visited = {start}
    parent = {start: None}
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
        start = tuple(np.random.randint(0, size, size=2))
        goal = tuple(np.random.randint(0, size, size=2))
        if env[start] == 1 or env[goal] == 1 or start == goal: continue
        env[start] = 0; env[goal] = 0
        if find_shortest_path_bfs(env, start, goal):
            print("성공: 풀 수 있는 맵을 생성했습니다.")
            return env, start, goal

# DQN 모델 관련 설정 및 클래스
actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
SIZE = 15
MAX_STEPS = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__(); self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1); self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1); self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1); self.bn3 = nn.BatchNorm2d(64)
        self.head = nn.Linear(w * h * 64, outputs)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

def get_state_tensor(env, agent_pos, goal_pos):
    obstacle_map = torch.from_numpy(env).float().unsqueeze(0); agent_map = torch.zeros_like(obstacle_map)
    if agent_pos: agent_map[0, agent_pos[0], agent_pos[1]] = 1
    goal_map = torch.zeros_like(obstacle_map)
    if goal_pos: goal_map[0, goal_pos[0], goal_pos[1]] = 1
    return torch.cat([obstacle_map, agent_map, goal_map], dim=0).unsqueeze(0).to(device)

# ------------------------------------------------------------------
# 2. GUI 애플리케이션 클래스
# ------------------------------------------------------------------
class PathfindingGUI:
    def __init__(self, master):
        self.master = master
        master.title("DQN Pathfinding Simulator")

        # 모델 불러오기
        self.agent_model = self.load_model()
        if self.agent_model is None:
            master.destroy()
            return

        # Matplotlib Figure와 Tkinter Canvas 설정
        self.fig = plt.Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # 버튼들을 담을 프레임 생성
        button_frame = tk.Frame(master)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        # 버튼 생성
        self.btn_find_path = tk.Button(button_frame, text="경로 찾기", command=self.find_path)
        self.btn_find_path.pack(side=tk.LEFT, expand=True, padx=5)

        self.btn_new_map = tk.Button(button_frame, text="새로운 맵 생성", command=self.new_map)
        self.btn_new_map.pack(side=tk.LEFT, expand=True, padx=5)

        self.btn_quit = tk.Button(button_frame, text="종료", command=master.quit)
        self.btn_quit.pack(side=tk.LEFT, expand=True, padx=5)

        # 초기 맵 생성 및 그리기
        self.new_map()

    def load_model(self):
        """저장된 dqn_model.pt 파일을 불러옵니다."""
        try:
            model = DQN(SIZE, SIZE, len(actions)).to(device)
            model.load_state_dict(torch.load("dqn_model.pt", map_location=device))
            model.eval() # 평가 모드로 설정
            print("성공: 'dqn_model.pt' 모델을 불러왔습니다.")
            return model
        except FileNotFoundError:
            messagebox.showerror("오류", "'dqn_model.pt' 파일을 찾을 수 없습니다. 스크립트와 같은 폴더에 있는지 확인하세요.")
            return None
        except Exception as e:
            messagebox.showerror("오류", f"모델을 불러오는 중 오류가 발생했습니다: {e}")
            return None

    def new_map(self):
        """새로운 맵을 생성하고 화면에 그립니다."""
        self.env, self.start, self.goal = generate_map_with_guaranteed_path(size=SIZE)
        self.draw_map()

    def find_path(self):
        """DQN 모델을 이용해 현재 맵에서 경로를 찾고 그립니다."""
        print("훈련된 모델로 경로를 탐색합니다...")
        state = self.start
        agent_path = [state]
        for _ in range(MAX_STEPS):
            state_tensor = get_state_tensor(self.env, state, self.goal)
            with torch.no_grad():
                action_idx = self.agent_model(state_tensor).max(1)[1].item()
            
            dx, dy = actions[action_idx]
            next_state = (state[0] + dx, state[1] + dy)
            
            if not (0 <= next_state[0] < SIZE and 0 <= next_state[1] < SIZE and self.env[next_state] == 0):
                print("탐색 중단: 정책이 벽으로 이동했습니다.")
                break
            
            state = next_state
            agent_path.append(state)
            if state == self.goal:
                print("탐색 성공: 목표에 도달했습니다!")
                break
        
        self.draw_map(path=agent_path) # 경로를 포함하여 다시 그리기

    def draw_map(self, path=None):
        """맵, 시작/도착점, 그리고 선택적으로 경로를 그립니다."""
        self.ax.clear()
        
        visual = np.stack([self.env.copy().astype(float)] * 3, axis=-1)
        
        # 경로가 있으면 노란색으로 그림
        if path:
            for pos in path:
                if pos != self.start and pos != self.goal:
                    visual[pos] = [1, 1, 0] # Yellow
        
        visual[self.start] = [0, 1, 0]  # 시작: 초록색
        visual[self.goal] = [0, 0, 1]   # 목표: 파란색
        
        self.ax.imshow(visual)
        self.ax.set_title("DQN Pathfinding Simulator")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self.canvas.draw()

# ------------------------------------------------------------------
# 3. 메인 실행 부분
# ------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = PathfindingGUI(root)
    root.mainloop()