# 파일명: dqn_trainer_final.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. 환경 및 기본 설정
# ------------------------------------------------------------------
def find_shortest_path_bfs(env, start, goal):
    """BFS를 사용하여 최단 경로가 존재하는지 확인하고, 있다면 경로를 반환합니다."""
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
            return path[::-1] # 경로가 존재하면 경로 리스트 반환

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and env[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
                parent[(nr, nc)] = (r, c)
                
    return [] # 경로가 없으면 빈 리스트 반환

def generate_map_with_guaranteed_path(size=15, obstacle_prob=0.3):
    """시작점에서 도착점까지 가는 길이 반드시 존재하는 맵을 생성합니다."""
    while True:
        env = np.random.choice([0, 1], size=(size, size), p=[1 - obstacle_prob, obstacle_prob])
        start = tuple(np.random.randint(0, size, size=2))
        goal = tuple(np.random.randint(0, size, size=2))
        if env[start] == 1 or env[goal] == 1 or start == goal:
            continue
        env[start] = 0
        env[goal] = 0
        if find_shortest_path_bfs(env, start, goal): # 경로 존재 여부 확인
            return env, start, goal

# --- 하이퍼파라미터 설정 ---
actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
SIZE = 15
ALPHA = 0.001
GAMMA = 0.99
BATCH_SIZE = 256
MEMORY_SIZE = 50000
TARGET_UPDATE = 10
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 40000   # 일반화 학습을 위해 더 긴 탐험 시간 보장
MAX_STEPS = 300
MAX_EPISODES = 30000    # 일반화 학습을 위해 더 많은 에피소드 진행

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# 2. DQN 핵심 구성 요소
# ------------------------------------------------------------------
class ReplayMemory:
    def __init__(self, capacity): self.memory = deque([], maxlen=capacity)
    def push(self, *args): self.memory.append(args)
    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self): return len(self.memory)

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1); self.bn1 = nn.BatchNorm2d(16)
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

class DQNAgent:
    def __init__(self):
        self.policy_net = DQN(SIZE, SIZE, len(actions)).to(device); self.target_net = DQN(SIZE, SIZE, len(actions)).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=ALPHA); self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0
    def select_action(self, state):
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad(): return self.policy_net(state).max(1)[1].view(1, 1)
        else: return torch.tensor([[random.randrange(len(actions))]], device=device, dtype=torch.long)
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

# ------------------------------------------------------------------
# 3. 메인 훈련 실행 부분
# ------------------------------------------------------------------
if __name__ == "__main__":
    agent = DQNAgent()
    episode_rewards = []
    
    print(f"사용 장치: {device}")
    print(f"\n일반화 모델 훈련 시작! 총 {MAX_EPISODES} 에피소드 진행")
    
    for i_episode in range(MAX_EPISODES):
        # 일반화 학습을 위해 매 에피소드마다 새로운 맵 생성
        env, start, goal = generate_map_with_guaranteed_path(size=SIZE)
        
        state = start
        total_reward = 0
        for t in range(MAX_STEPS):
            current_state_tensor = get_state_tensor(env, state, goal)
            action_tensor = agent.select_action(current_state_tensor)
            action_idx = action_tensor.item()
            dx, dy = actions[action_idx]
            next_state = (state[0] + dx, state[1] + dy)
            
            done = False
            is_wall_hit = not (0 <= next_state[0] < SIZE and 0 <= next_state[1] < SIZE and env[next_state] == 0)
            
            # ★★★★★ 핵심 수정: 보상 설계를 더 적극적으로 변경 ★★★★★
            dist_before = np.sqrt((state[0] - goal[0])**2 + (state[1] - goal[1])**2)
            dist_after = np.sqrt((next_state[0] - goal[0])**2 + (next_state[1] - goal[1])**2)

            if dist_after < dist_before:
                reward = 1.0  # 가까워지면 보상 대폭 상향 (+0.1 -> +1.0)
            else:
                reward = -0.05 # 멀어지면 페널티 대폭 하향 (-0.2 -> -0.05)
            
            if is_wall_hit:
                reward = -1       # 벽 충돌 페널티는 그대로 유지
                next_state = state
            elif next_state == goal:
                reward = 20      # 목표 도달 보상도 상향 (10 -> 20)
                done = True
            
            total_reward += reward
            reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)
            next_state_tensor = None if done else get_state_tensor(env, next_state, goal)
            agent.memory.push(current_state_tensor, action_tensor, reward_tensor, next_state_tensor)
            
            state = next_state
            
            agent.optimize_model()
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        if (i_episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"훈련 진행... Episode {i_episode+1}/{MAX_EPISODES}, Avg Reward (last 100): {avg_reward:.2f}")

        if i_episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
    print("훈련 완료!")
    
    # 훈련된 모델을 .pt 파일로 저장
    MODEL_PATH = "dqn_model_generalized.pt"
    torch.save(agent.policy_net.state_dict(), MODEL_PATH)
    print(f"\n일반화 훈련된 모델이 '{MODEL_PATH}' 경로에 저장되었습니다.")

    # 보상 그래프 시각화
    plt.figure(); plt.title('Training Rewards (Generalization)'); plt.xlabel('Episode'); plt.ylabel('Total Reward')
    plt.plot(episode_rewards); plt.savefig('dqn_rewards_generalized.png'); plt.show()