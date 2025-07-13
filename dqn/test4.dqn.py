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
    """BFS를 사용하여 최단 경로를 찾아 좌표 리스트로 반환합니다."""
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
    """시작점에서 도착점까지 가는 길이 반드시 존재하는 맵을 생성합니다."""
    print("\n경로가 보장된 새로운 맵을 생성하는 중...")
    while True:
        env = np.random.choice([0, 1], size=(size, size), p=[1-obstacle_prob, obstacle_prob])
        start = tuple(np.random.randint(0, size, size=2))
        goal = tuple(np.random.randint(0, size, size=2))
        if env[start] == 1 or env[goal] == 1 or start == goal: continue
        env[start] = 0; env[goal] = 0
        if find_shortest_path_bfs(env, start, goal):
            print("성공: 풀 수 있는 맵을 생성했습니다.")
            return env, start, goal

actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
SIZE = 15; ALPHA = 0.001; GAMMA = 0.99; BATCH_SIZE = 256; MEMORY_SIZE = 50000
TARGET_UPDATE = 10; EPSILON_START = 1.0; EPSILON_END = 0.01; EPSILON_DECAY = 20000
MAX_STEPS = 300; MAX_EPISODES = 15000
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

class DQNAgent:
    def __init__(self):
        self.policy_net = DQN(SIZE, SIZE, len(actions)).to(device); self.target_net = DQN(SIZE, SIZE, len(actions)).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=ALPHA); self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0
    def select_action(self, state, use_exploration=True):
        if use_exploration:
            eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * self.steps_done / EPSILON_DECAY)
            self.steps_done += 1
            if random.random() > eps_threshold:
                with torch.no_grad(): return self.policy_net(state).max(1)[1].view(1, 1)
            else: return torch.tensor([[random.randrange(len(actions))]], device=device, dtype=torch.long)
        else:
            with torch.no_grad(): return self.policy_net(state).max(1)[1].view(1, 1)
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
# 3. 메인 실행 부분
# ------------------------------------------------------------------
if __name__ == "__main__":
    print(f"사용 장치: {device}")
    
    # ======================================================================
    # 파트 1: 훈련
    # ======================================================================
    trained_agent = DQNAgent()
    episode_rewards = []
    train_env, train_start, train_goal = generate_map_with_guaranteed_path(size=SIZE)
    
    print("생성된 '훈련용' 맵을 확인하세요. 창을 닫으면 학습이 시작됩니다.")
    plt.figure(figsize=(6, 6)); map_visual = np.stack([train_env.copy().astype(float)] * 3, axis=-1)
    map_visual[train_start] = [0, 1, 0]; map_visual[train_goal] = [0, 0, 1]
    plt.imshow(map_visual); plt.title("Training Map - Press [X] to Start"); plt.axis('off'); plt.show()
    
    print(f"\n훈련 시작! 시작점: {train_start}, 목표점: {train_goal}")
    for i_episode in range(MAX_EPISODES):
        state = train_start
        total_reward = 0
        for t in range(MAX_STEPS):
            current_state_tensor = get_state_tensor(train_env, state, train_goal)
            action_tensor = trained_agent.select_action(current_state_tensor)
            action_idx = action_tensor.item()
            dx, dy = actions[action_idx]; next_state = (state[0] + dx, state[1] + dy)
            done = False
            dist_before = np.sqrt((state[0] - train_goal[0])**2 + (state[1] - train_goal[1])**2)
            dist_after = np.sqrt((next_state[0] - train_goal[0])**2 + (next_state[1] - train_goal[1])**2)
            if dist_after < dist_before: reward = 0.1
            else: reward = -0.2
            if not (0 <= next_state[0] < SIZE and 0 <= next_state[1] < SIZE and train_env[next_state] == 0):
                reward = -1; done = True
            elif next_state == train_goal:
                reward = 10; done = True
            total_reward += reward
            reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)
            next_state_tensor = None if done else get_state_tensor(train_env, next_state, train_goal)
            trained_agent.memory.push(current_state_tensor, action_tensor, reward_tensor, next_state_tensor)
            if not done: state = next_state
            trained_agent.optimize_model()
            if done: break
        
        episode_rewards.append(total_reward)
        if (i_episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"훈련 진행... Episode {i_episode+1}/{MAX_EPISODES}, Avg Reward (last 100): {avg_reward:.2f}")
        if i_episode % TARGET_UPDATE == 0:
            trained_agent.target_net.load_state_dict(trained_agent.policy_net.state_dict())
            
    print("훈련 완료!")
    
    # ★★★★★ 새로운 기능 1: 훈련된 모델의 가중치(state_dict)를 파일로 저장 ★★★★★
    MODEL_PATH = "dqn_model.pt"
    torch.save(trained_agent.policy_net.state_dict(), MODEL_PATH)
    print(f"\n훈련된 모델이 '{MODEL_PATH}' 경로에 저장되었습니다.")

    # --- (참고) 저장된 모델 불러오는 방법 ---
    # model_to_load = DQN(SIZE, SIZE, len(actions)).to(device)
    # model_to_load.load_state_dict(torch.load(MODEL_PATH))
    # model_to_load.eval()
    
    # ======================================================================
    # 파트 2: 테스트
    # ======================================================================
    NUM_TESTS = 5
    print(f"\n{'='*25}\n저장된 모델로 {NUM_TESTS}개의 새로운 맵 테스트 시작\n{'='*25}")

    for i in range(NUM_TESTS):
        print(f"\n--- 테스트 맵 #{i+1} ---")
        test_env, test_start, test_goal = generate_map_with_guaranteed_path(size=SIZE)
        shortest_path = find_shortest_path_bfs(test_env, test_start, test_goal)
        agent_path = [test_start]
        state = test_start
        for _ in range(MAX_STEPS * 2):
            state_tensor = get_state_tensor(test_env, state, test_goal)
            action_tensor = trained_agent.select_action(state_tensor, use_exploration=False)
            action_idx = action_tensor.item()
            dx, dy = actions[action_idx]; next_state = (state[0] + dx, state[1] + dy)
            if not (0 <= next_state[0] < SIZE and 0 <= next_state[1] < SIZE and test_env[next_state] == 0):
                print("테스트 중단: 에이전트가 벽으로 이동했습니다.")
                break
            state = next_state; agent_path.append(state)
            if state == test_goal:
                print("테스트 성공: 에이전트가 목표에 도달했습니다!")
                break
        
        print(f"최단 경로 길이 (BFS): {len(shortest_path) - 1 if shortest_path else 'N/A'}")
        if agent_path[-1] == test_goal: print(f"에이전트 경로 길이: {len(agent_path) - 1}")
        else: print("에이전트가 목표에 도달하지 못했습니다.")
            
        plt.figure(figsize=(8, 8)); visual_shortest = np.stack([test_env.copy().astype(float)] * 3, axis=-1)
        for pos in shortest_path:
            if pos != test_start and pos != test_goal: visual_shortest[pos] = [1, 0, 1]
        visual_shortest[test_start] = [0, 1, 0]; visual_shortest[test_goal] = [0, 0, 1]
        plt.imshow(visual_shortest); plt.title(f"Test #{i+1} - Shortest Path (BFS)"); plt.axis('off'); plt.show()
        plt.figure(figsize=(8, 8)); visual_agent = np.stack([test_env.copy().astype(float)] * 3, axis=-1)
        for pos in agent_path:
            if pos != test_start and pos != test_goal: visual_agent[pos] = [1, 1, 0]
        visual_agent[test_start] = [0, 1, 0]; visual_agent[test_goal] = [0, 0, 1]
        plt.imshow(visual_agent); plt.title(f"Test #{i+1} - Agent's Path (DQN)"); plt.axis('off'); plt.show()
        plt.figure(figsize=(8, 8)); visual_combined = np.stack([test_env.copy().astype(float)] * 3, axis=-1)
        for pos in shortest_path: visual_combined[pos] = [1, 0, 1]
        for pos in agent_path: visual_combined[pos] = [1, 1, 0]
        visual_combined[test_start] = [0, 1, 0]; visual_combined[test_goal] = [0, 0, 1]
        plt.imshow(visual_combined); plt.title(f"Test #{i+1} - Shortest(M) vs Agent(Y)"); plt.axis('off'); plt.show()