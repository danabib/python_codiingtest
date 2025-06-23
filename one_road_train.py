import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from collections import deque

def generate_map(size=50, obstacle_prob=0.2):
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

# 하이퍼파라미터
size = 50
alpha = 0.1
gamma = 0.9
max_steps = 2000
same_pos_threshold = 20
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.01

# Q 테이블 초기화
Q = np.zeros((size, size, 4))

# 학습 루프 설정
plt.figure(figsize=(8, 8))
episode = 0

CONVERGENCE_CRITERIA_WINDOW = 100
CONVERGENCE_THRESHOLD = 0.98
MAX_EPISODES = 20000
recent_outcomes = deque(maxlen=CONVERGENCE_CRITERIA_WINDOW)

env, start, goal = generate_map(size=size)
print(f"학습을 시작할 {size}x{size} 고정된 맵이 생성되었습니다.")
print(f"시작점: {start}, 목표점: {goal}")

# ★★★★★ 1. 마지막 성공 경로를 저장할 변수 초기화 ★★★★★
last_successful_path = []

while episode < MAX_EPISODES:
    state = start
    succeeded = False
    
    # ★★★★★ 2. 매 에피소드의 경로를 기록할 리스트 초기화 ★★★★★
    current_path = [state]

    for step in range(max_steps):
        x, y = state
        if np.random.rand() < epsilon:
            action = np.random.choice(list(actions.keys()))
        else:
            action = np.argmax(Q[x, y])
        
        dx, dy = actions[action]
        nx, ny = x + dx, y + dy

        if 0 <= nx < size and 0 <= ny < size and env[nx, ny] == 0:
            next_state = (nx, ny)
        else:
            next_state = state
        
        reward = -1
        if next_state == goal: reward = 100
        elif next_state == state: reward = -10
        
        Q[x, y, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[x, y, action])
        
        state = next_state
        current_path.append(state) # 이동할 때마다 경로 기록
        
        if state == goal:
            succeeded = True
            break
            
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # ★★★★★ 3. 에피소드 성공 시, 마지막 성공 경로 업데이트 ★★★★★
    if succeeded:
        last_successful_path = list(current_path) # 리스트를 복사해서 저장
        print(f"Episode {episode+1}: 성공! (경로 길이: {len(last_successful_path)}, e: {epsilon:.3f})")
    else:
        # 실패 시에는 메시지만 출력
        print(f"Episode {episode+1}: 실패. (e: {epsilon:.3f})")
        
    episode += 1
    recent_outcomes.append(1 if succeeded else 0)

    if len(recent_outcomes) == CONVERGENCE_CRITERIA_WINDOW and np.mean(list(recent_outcomes)) >= CONVERGENCE_THRESHOLD:
        print(f"\n성공률이 {CONVERGENCE_THRESHOLD*100}%에 도달하여 학습을 종료합니다.")
        break

if episode == MAX_EPISODES:
    print(f"\n최대 에피소드 {MAX_EPISODES}에 도달하여 학습을 종료합니다.")

print("\n최종 학습이 완료되었습니다.")
plt.close()

# ======================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
#                  최종 경로 시각화 (저장된 경로를 사용하도록 변경)
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ======================================================================

print("\n마지막으로 성공한 실제 경로를 시각화합니다.")

plt.figure(figsize=(8, 8))
final_visual = np.stack([env.copy().astype(float)] * 3, axis=-1)

if not last_successful_path:
    print("학습 중 한 번도 성공하지 못해 표시할 경로가 없습니다.")
else:
    print(f"표시할 경로의 길이: {len(last_successful_path)}")
    # 저장된 마지막 성공 경로를 노란색으로 칠하기
    for pos in last_successful_path:
        if pos != start and pos != goal:
            final_visual[pos] = [1, 1, 0] # 경로: 노란색

# 시작점(초록)과 도착점(파란)을 마지막에 그려서 경로 위에 표시
final_visual[start] = [0, 1, 0]
final_visual[goal] = [0, 0, 1]

plt.imshow(final_visual)
plt.title("Last Successful Path")
plt.axis('off')
plt.show()