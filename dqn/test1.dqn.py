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
def generate_map(size=20, obstacle_prob=0.3):
    env = np.random.choice([0, 1], size=(size, size), p=[1-obstacle_prob, obstacle_prob])
    start = tuple(np.random.randint(0, size, size=2))
    goal = tuple(np.random.randint(0, size, size=2))
    while env[start] == 1 or env[goal] == 1 or start == goal:
        start = tuple(np.random.randint(0, size, size=2))
        goal = tuple(np.random.randint(0, size, size=2))
    env[start] = 0
    env[goal] = 0
    return env, start, goal

actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

# ======================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 하이퍼파라미터 설명 추가 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ======================================================================
# --- 환경 관련 ---
SIZE = 20                 # 맵의 가로, 세로 크기

# --- DQN 학습 관련 ---
ALPHA = 0.001             # 학습률 (Learning Rate): 모델을 한 번에 얼마나 업데이트할지 결정하는 보폭 크기
GAMMA = 0.99              # 할인율 (Discount Factor): 미래에 받을 보상을 현재 가치로 환산할 때 얼마나 할인할지 결정 (1에 가까울수록 미래를 더 중요하게 생각)
BATCH_SIZE = 128          # 배치 크기: 메모리에서 한 번에 몇 개의 경험을 꺼내 학습할지 결정
MEMORY_SIZE = 30000       # 리플레이 메모리 크기: 에이전트가 기억할 수 있는 총 경험의 수
TARGET_UPDATE = 10        # 타겟 네트워크 업데이트 주기: 몇 에피소드마다 타겟 네트워크를 업데이트할지 결정

# --- 탐험(Exploration) 관련 ---
EPSILON_START = 1.0       # 초기 엡실론: 학습 시작 시 무작위 행동을 할 확률 (100%에서 시작)
EPSILON_END = 0.01        # 최종 엡실론: 학습 막바지에 도달할 최소 무작위 행동 확률 (최소 1%는 탐험)
EPSILON_DECAY = 10000     # 엡실론 감쇠율: 엡실론이 얼마나 빠르게 감소할지 결정. 클수록 천천히 감소.

# --- 에피소드 진행 관련 ---
MAX_STEPS = 500           # 한 에피소드당 최대 스텝 수: 에이전트가 한 판에 움직일 수 있는 최대 횟수
MAX_EPISODES = 5000       # 총 학습 에피소드 수: 에이전트가 총 몇 판을 플레이하며 학습할지 결정

# GPU 사용 설정 (자동 감지)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------------
# 2. DQN 핵심 구성 요소
# ------------------------------------------------------------------
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(args)
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        linear_input_size = w * h * 64
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

def get_state_tensor(env, agent_pos, goal_pos):
    obstacle_map = torch.tensor(env, dtype=torch.float32).unsqueeze(0)
    agent_map = torch.zeros_like(obstacle_map)
    if agent_pos is not None:
        agent_map[0, agent_pos[0], agent_pos[1]] = 1
    goal_map = torch.zeros_like(obstacle_map)
    if goal_pos is not None:
        goal_map[0, goal_pos[0], goal_pos[1]] = 1
    return torch.cat([obstacle_map, agent_map, goal_map], dim=0).unsqueeze(0).to(device)

# ------------------------------------------------------------------
# 3. DQN 에이전트 클래스
# ------------------------------------------------------------------
class DQNAgent:
    def __init__(self):
        self.policy_net = DQN(SIZE, SIZE, len(actions)).to(device)
        self.target_net = DQN(SIZE, SIZE, len(actions)).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=ALPHA)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0

    def select_action(self, state):
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                        np.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(len(actions))]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = list(zip(*transitions))

        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[2])
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch[3])), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch[3] if s is not None]) if any(non_final_mask) else None

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        if non_final_next_states is not None:
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

# ------------------------------------------------------------------
# 4. 메인 학습 루프
# ------------------------------------------------------------------
if __name__ == "__main__":
    agent = DQNAgent()
    episode_rewards = []
    
    for i_episode in range(MAX_EPISODES):
        env, start, goal = generate_map(size=SIZE)
        state = start
        
        total_reward = 0
        for t in range(MAX_STEPS):
            current_state_tensor = get_state_tensor(env, state, goal)
            action_tensor = agent.select_action(current_state_tensor)
            action_idx = action_tensor.item()
            
            dx, dy = actions[action_idx]
            next_state = (state[0] + dx, state[1] + dy)
            
            reward = -0.1
            done = False
            
            if not (0 <= next_state[0] < SIZE and 0 <= next_state[1] < SIZE and env[next_state] == 0):
                reward = -1
                done = True
            elif next_state == goal:
                reward = 10
                done = True

            total_reward += reward
            reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)
            
            if done:
                next_state_tensor = None
            else:
                next_state_tensor = get_state_tensor(env, next_state, goal)

            agent.memory.push(current_state_tensor, action_tensor, reward_tensor, next_state_tensor)
            
            if not done:
                state = next_state
            
            agent.optimize_model()
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        if (i_episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {i_episode+1}/{MAX_EPISODES}, Avg Reward (last 100): {avg_reward:.2f}")

        if i_episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
    print("훈련 완료")
    
    # 보상 그래프 그리기
    plt.figure()
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(episode_rewards)
    plt.savefig('dqn_rewards.png') # 그래프를 파일로 저장
    plt.show()