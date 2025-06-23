import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from collections import deque # deque 임포트 확인

# 환경 생성 함수
def generate_map(size=10, obstacle_prob=0.2):
    env = np.random.choice([0, 1], size=(size, size), p=[1-obstacle_prob, obstacle_prob])
    start = tuple(np.random.randint(0, size, size=2))
    goal = tuple(np.random.randint(0, size, size=2))

    while env[start] == 1 or env[goal] == 1 or start == goal:
        start = tuple(np.random.randint(0, size, size=2))
        goal = tuple(np.random.randint(0, size, size=2))

    env[start] = 0
    env[goal] = 0
    return env, start, goal

# 이동 방향 정의
actions = {
    0: (-1, 0),  # 위
    1: (1, 0),   # 아래
    2: (0, -1),  # 왼쪽
    3: (0, 1)    # 오른쪽
}

# 하이퍼파라미터
size = 10
alpha = 0.1 #목적지를 찾았을 때, 그 값을 곱해서 수식으로 값을 대체 하는거
gamma = 0.9 #
epsilon = 0.1 #새로운 지역으로 이동할 확률(새로운 탐험)
max_steps = 100 #무한루프를 하지 않도록 한 에피소드에 100번까지 시도하고 안되면 다음으로 넘어감
same_pos_threshold = 10  

# Q 테이블 초기화
Q = np.zeros((size, size, 4))

# 시각화 함수 (수정 완료된 버전)
def render(env, pos, start, goal, episode, total_attempt):
    visual = np.stack([env.copy().astype(float)] * 3, axis=-1)
    
    sx, sy = start
    gx, gy = goal
    px, py = pos

    visual[sx, sy] = [0, 1, 0]  # 시작 위치 (Green)
    visual[gx, gy] = [0, 0, 1]  # 목표 위치 (Blue)
    visual[px, py] = [1, 0, 0]  # 현재 위치 (Red)

    plt.imshow(visual)
    plt.title(f"Episode {episode+1} - Agent Moving...") 
    plt.axis('off')
    plt.pause(0.01) 
    plt.clf()


# 학습 루프
plt.figure(figsize=(5, 5))
episode = 0

CONVERGENCE_CRITERIA_WINDOW = 50  # 최근 50개의 에피소드를 기준으로 판단
CONVERGENCE_THRESHOLD = 0.8      # 성공률이 80% 이상일 때
MAX_EPISODES = 5000               # 무한 루프 방지를 위한 최대 시도 횟수
recent_outcomes = deque(maxlen=CONVERGENCE_CRITERIA_WINDOW) # 최근 결과 저장

# episode 기반의 새로운 메인 루프
while episode < MAX_EPISODES:
    env, start, goal = generate_map(size=size)
    state = start
    visited_counter = defaultdict(int)
    visited_counter[state] += 1
    
    succeeded = False # 이번 에피소드 성공 여부 플래그

    for step in range(max_steps):
        x, y = state
        
        # 행동 선택 (Epsilon-Greedy)
        if np.random.rand() < epsilon:
            action = np.random.choice(list(actions.keys()))
        else:
            action = np.argmax(Q[x, y])
            
        dx, dy = actions[action]
        nx, ny = x + dx, y + dy

        # 상태 전이 및 보상
        if 0 <= nx < size and 0 <= ny < size and env[nx, ny] == 0:
            next_state = (nx, ny)
        else:
            next_state = state # 벽에 부딪히거나 맵 밖으로 나가면 제자리
        
        if next_state == goal:
            reward = 100
        elif next_state == state: # 제자리에 머무른 경우 (벽)
            reward = -10
        else: # 일반 이동
            reward = -1
        
        # Q-러닝 업데이트
        Q[x, y, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[x, y, action])
        
        # 시각화 (render 함수 호출 시 episode 변수 사용)
        render(env, state, start, goal, episode, episode + 1)

        # 목표 도달 시
        if next_state == goal:
            print(f"[Episode {episode+1}] 목표 도달! (Steps: {step})")
            succeeded = True
            break # 내부 for 루프 탈출
        
        # 같은 위치 반복으로 인한 종료
        visited_counter[next_state] += 1
        if visited_counter[next_state] >= same_pos_threshold:
            # print(f"[Episode {episode+1}] 도달 불가능. (반복 위치: {next_state})")
            succeeded = False
            break # 내부 for 루프 탈출

        state = next_state
    
    # --- 매 에피소드 종료 후 수렴 여부 체크 ---
    episode += 1
    recent_outcomes.append(1 if succeeded else 0)

    # 최근 결과가 충분히 쌓이면 성공률을 계산하여 수렴 여부 판단
    if len(recent_outcomes) == CONVERGENCE_CRITERIA_WINDOW:
        success_rate = np.mean(list(recent_outcomes))
        print(f"Episode {episode}: 최근 {CONVERGENCE_CRITERIA_WINDOW}번 성공률: {success_rate:.2f}")
        if success_rate >= CONVERGENCE_THRESHOLD:
            print(f"\n성공률이 {CONVERGENCE_THRESHOLD*100}%에 도달하여 학습을 종료합니다.")
            break

# 학습 종료 메시지
if episode == MAX_EPISODES:
    print(f"\n최대 에피소드 {MAX_EPISODES}에 도달하여 학습을 종료합니다.")

print("\n최종 학습이 완료되었습니다.")
plt.close()