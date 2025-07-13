# 파일명: dqn_trainer_ultimate.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import math

# --- 환경 및 기본 설정 ---
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
    while True:
        env = np.random.choice([0, 1], size=(size, size), p=[1 - obstacle_prob, obstacle_prob])
        start = tuple(np.random.randint(0, size, size=2)); goal = tuple(np.random.randint(0, size, size=2))
        if env[start] == 1 or env[goal] == 1 or start == goal: continue
        env[start] = 0; env[goal] = 0
        if find_shortest_path_bfs(env, start, goal):
            return env, start, goal

actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

# ★★★★★ 1시간 내 훈련을 위한 하이퍼파라미터 최적화 ★★★★★
SIZE = 10
ALPHA = 0.001
GAMMA = 0.99
BATCH_SIZE = 256
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
EPSILON_START = 1.0
EPSILON_END = 0.05 # 탐험의 최소값 약간 증가
EPSILON_DECAY = 4000 # 엡실론 감쇠 속도 가속
MAX_STEPS = 150
MAX_EPISODES = 2500 # 전체 훈련 시간 단축

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DQN 핵심 구성 요소 ---
class ReplayMemory:
    def __init__(self, capacity): self.memory = deque([], maxlen=capacity)
    def push(self, *args): self.memory.append(args)
    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self): return len(self.memory)

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
    dx = (goal_pos[1] - agent_pos[1]) / (size-1); dy = (goal_pos[0] - agent_pos[0]) / (size-1)
    dx_map = torch.full_like(obstacle_map, dx); dy_map = torch.full_like(obstacle_map, dy)
    return torch.cat([obstacle_map, agent_map, goal_map, dx_map, dy_map], dim=0).unsqueeze(0).to(device)

class DQNAgent:
    def __init__(self, size, outputs):
        self.policy_net = DQN(size, size, outputs).to(device); self.target_net = DQN(size, size, outputs).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=ALPHA); self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0

    # ★★★★★ 핵심 수정: '지능적인 탐험' 로직 추가 ★★★★★
    def select_action(self, state_tensor, current_pos, goal_pos, env):
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        
        # 1. 활용(Exploitation): 학습된 최적의 행동 선택
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state_tensor).max(1)[1].item()
        # 2. 탐험(Exploration)
        else:
            # 50% 확률로 완전 무작위 탐험
            if random.random() < 0.5:
                return random.randrange(len(actions))
            # 50% 확률로 목표 방향으로 가는 '지능적 탐험'
            else:
                best_action = -1
                min_dist = float('inf')
                # 현재 위치에서 모든 방향으로 가보고, 목표와 가장 가까워지는 행동을 선택
                for i, move in actions.items():
                    next_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
                    if (0 <= next_pos[0] < SIZE and 0 <= next_pos[1] < SIZE and env[next_pos] == 0):
                        dist = np.sqrt((next_pos[0] - goal_pos[0])**2 + (next_pos[1] - goal_pos[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_action = i
                return best_action if best_action != -1 else random.randrange(len(actions))

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE: return
        transitions = self.memory.sample(BATCH_SIZE); batch = list(zip(*transitions))
        state_batch = torch.cat(batch[0]); action_batch = torch.cat(batch[1]); reward_batch = torch.cat(batch[2])
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch[3])), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch[3] if s is not None]) if any(non_final_mask) else None
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        if non_final_next_states is not None:
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100); self.optimizer.step()

# --- 메인 훈련 실행 부분 ---
if __name__ == "__main__":
    agent = DQNAgent(SIZE, len(actions))
    episode_rewards = []
    print(f"사용 장치: {device}")
    print(f"\n최종 고속 일반화 모델 훈련 시작! 총 {MAX_EPISODES} 에피소드 진행 (맵 크기: {SIZE}x{SIZE})")
    
    for i_episode in range(MAX_EPISODES):
        env, start, goal = generate_map_with_guaranteed_path(size=SIZE)
        state = start
        total_reward = 0
        for t in range(MAX_STEPS):
            current_state_tensor = get_state_tensor(env, state, goal)
            action_idx = agent.select_action(current_state_tensor, state, goal, env) # 지능적 탐험에 필요한 정보 전달
            
            dx, dy = actions[action_idx]
            next_state = (state[0] + dx, state[1] + dy)
            
            done = False
            is_wall_hit = not (0 <= next_state[0] < SIZE and 0 <= next_state[1] < SIZE and env[next_state] == 0)
            
            if is_wall_hit:
                reward = -1
                next_state = state
            elif next_state == goal:
                reward = 20
                done = True
            else:
                reward = -0.05
            
            total_reward += reward
            action_tensor = torch.tensor([[action_idx]], device=device)
            reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)
            next_state_tensor = None if done else get_state_tensor(env, next_state, goal)
            agent.memory.push(current_state_tensor, action_tensor, reward_tensor, next_state_tensor)
            
            state = next_state
            agent.optimize_model()
            
            if done: break
        
        episode_rewards.append(total_reward)
        if (i_episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"훈련 진행... Episode {i_episode+1}/{MAX_EPISODES}, Avg Reward (last 100): {avg_reward:.2f}")

        if i_episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
    print("훈련 완료!")
    MODEL_PATH = "dqn_model_ultimate.pt"
    torch.save(agent.policy_net.state_dict(), MODEL_PATH)
    print(f"\n최종 훈련된 모델이 '{MODEL_PATH}' 경로에 저장되었습니다.")
    
    plt.figure(); plt.title('Ultimate Training Rewards'); plt.xlabel('Episode'); plt.ylabel('Total Reward')
    plt.plot(episode_rewards); plt.savefig('dqn_rewards_ultimate.png'); plt.show()