# 파일명: ppo_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

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

def generate_map_with_guaranteed_path(size=10, obstacle_prob=0.25):
    while True:
        env = np.random.choice([0, 1], size=(size, size), p=[1 - obstacle_prob, obstacle_prob])
        start = tuple(np.random.randint(0, size, size=2)); goal = tuple(np.random.randint(0, size, size=2))
        if env[start] == 1 or env[goal] == 1 or start == goal: continue
        env[start] = 0; env[goal] = 0
        if find_shortest_path_bfs(env, start, goal):
            return env, start, goal

actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
SIZE = 10; ALPHA = 0.0003; GAMMA = 0.99; PPO_EPSILON = 0.2
K_EPOCHS = 1; T_HORIZON = 64; MAX_EPISODES = 2000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PPO 핵심 구성 요소: Actor-Critic 신경망 ---
class ActorCritic(nn.Module):
    def __init__(self, h, w, outputs):
        super(ActorCritic, self).__init__()
        # 몸통: 특징 추출부
        self.conv1 = nn.Conv2d(5, 16, kernel_size=3, padding=1); self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1); self.bn2 = nn.BatchNorm2d(32)
        
        # 머리 1: 액터 (정책 출력)
        self.actor_head = nn.Linear(w * h * 32, outputs)
        # 머리 2: 크리틱 (가치 출력)
        self.critic_head = nn.Linear(w * h * 32, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        
        # 행동 확률과 상태 가치를 각각 반환
        action_probs = F.softmax(self.actor_head(x), dim=-1)
        state_values = self.critic_head(x)
        return action_probs, state_values

def get_state_tensor(env, agent_pos, goal_pos):
    size = env.shape[0]
    obstacle_map = torch.from_numpy(env).float().unsqueeze(0); agent_map = torch.zeros_like(obstacle_map)
    if agent_pos: agent_map[0, agent_pos[0], agent_pos[1]] = 1
    goal_map = torch.zeros_like(obstacle_map)
    if goal_pos: goal_map[0, goal_pos[0], goal_pos[1]] = 1
    dx = (goal_pos[1] - agent_pos[1]) / (size - 1); dy = (goal_pos[0] - agent_pos[0]) / (size - 1)
    dx_map = torch.full_like(obstacle_map, dx); dy_map = torch.full_like(obstacle_map, dy)
    return torch.cat([obstacle_map, agent_map, goal_map, dx_map, dy_map], dim=0).unsqueeze(0).to(device)

# --- 메인 훈련 실행 부분 ---
if __name__ == "__main__":
    model = ActorCritic(SIZE, SIZE, len(actions)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=ALPHA)
    episode_rewards = []
    
    print(f"사용 장치: {device}")
    print(f"\nPPO 모델 훈련 시작! 총 {MAX_EPISODES} 에피소드 진행")

    for i_episode in range(MAX_EPISODES):
        env, start, goal = generate_map_with_guaranteed_path(size=SIZE)
        state = start
        done = False
        total_reward = 0
        
        # PPO는 일정 기간의 데이터를 모아서 한 번에 학습
        memory = []
        
        while not done:
            for t in range(T_HORIZON):
                state_tensor = get_state_tensor(env, state, goal)
                # 액터의 확률에 따라 행동 선택
                probs, _ = model(state_tensor)
                dist = Categorical(probs)
                action = dist.sample()
                action_idx = action.item()

                dx, dy = actions[action_idx]
                next_state = (state[0] + dx, state[1] + dy)
                
                is_wall_hit = not (0 <= next_state[0] < SIZE and 0 <= next_state[1] < SIZE and env[next_state] == 0)
                if is_wall_hit: reward = -1; next_state = state
                elif next_state == goal: reward = 20; done = True
                else: reward = -0.05
                
                memory.append((state_tensor, action, reward, done))
                state = next_state
                total_reward += reward
                if done: break
            
            # --- 데이터를 모은 후, 학습 시작 ---
            states_b, actions_b, rewards_b, dones_b = [], [], [], []
            for s, a, r, d in memory:
                states_b.append(s); actions_b.append(a)
                rewards_b.append(torch.tensor([r], dtype=torch.float, device=device))
                dones_b.append(torch.tensor([1-d], dtype=torch.float, device=device))
            
            # GAE (Generalized Advantage Estimation) 대신 간단한 할인된 보상 계산
            discounted_rewards = []
            R = 0
            for r in reversed(rewards_b):
                R = r + GAMMA * R
                discounted_rewards.insert(0, R)
            discounted_rewards = torch.cat(discounted_rewards).detach()

            # 여러 Epoch 동안 수집된 데이터로 학습
            for _ in range(K_EPOCHS):
                probs, state_values = model(torch.cat(states_b))
                dist = Categorical(probs)
                
                # PPO의 핵심 손실 함수 계산
                advantage = discounted_rewards - state_values.squeeze()
                ratio = torch.exp(dist.log_prob(torch.cat(actions_b)) - dist.log_prob(torch.cat(actions_b)).detach())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-PPO_EPSILON, 1+PPO_EPSILON) * advantage
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.smooth_l1_loss(state_values.squeeze(), discounted_rewards)
                
                loss = actor_loss + 0.5 * critic_loss # 두 손실을 합쳐서 최적화
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            memory = [] # 메모리 비우기
            if done: break
            
        episode_rewards.append(total_reward)
        if (i_episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"훈련 진행... Episode {i_episode+1}/{MAX_EPISODES}, Avg Reward (last 100): {avg_reward:.2f}")

    print("훈련 완료!")
    MODEL_PATH = "ppo_model.pt"
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nPPO 훈련된 모델이 '{MODEL_PATH}' 경로에 저장되었습니다.")
    
    plt.figure(); plt.title('PPO Training Rewards'); plt.xlabel('Episode'); plt.ylabel('Total Reward')
    plt.plot(episode_rewards); plt.savefig('ppo_rewards.png'); plt.show()