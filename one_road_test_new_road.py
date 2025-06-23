import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from collections import deque

# ------------------------------------------------------------------
# 기본 함수 및 설정
# ------------------------------------------------------------------

def generate_map(size=50, obstacle_prob=0.2):
    """지정된 크기와 장애물 확률로 무작위 맵을 생성합니다."""
    env = np.random.choice([0, 1], size=(size, size), p=[1-obstacle_prob, obstacle_prob])
    start = tuple(np.random.randint(0, size, size=2))
    goal = tuple(np.random.randint(0, size, size=2))
    while env[start] == 1 or env[goal] == 1 or start == goal:
        start = tuple(np.random.randint(0, size, size=2))
        goal = tuple(np.random.randint(0, size, size=2))
    env[start] = 0
    env[goal] = 0
    return env, start, goal

# 행동 정의
actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

# 하이퍼파라미터 (상수 형태로 관리)
SIZE = 50
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
MAX_STEPS = 2000
MAX_EPISODES = 10000
CONVERGENCE_CRITERIA_WINDOW = 100
CONVERGENCE_THRESHOLD = 0.98

# ------------------------------------------------------------------
# 하나의 맵에 대한 학습 및 평가 함수
# ------------------------------------------------------------------
def train_on_single_map(map_index):
    """하나의 맵에 대해 Q-러닝을 처음부터 실행하고 결과를 반환합니다."""
    
    # 새로운 맵과 빈 Q-테이블로 학습 시작
    env, start, goal = generate_map(size=SIZE)
    Q = np.zeros((SIZE, SIZE, 4))
    
    print(f"\n===== 테스트 맵 #{map_index} 학습 시작 =====")
    print(f"시작점: {start}, 목표점: {goal}")

    episode = 0
    recent_outcomes = deque(maxlen=CONVERGENCE_CRITERIA_WINDOW)
    is_converged = False

    while episode < MAX_EPISODES:
        state = start
        succeeded = False
        for step in range(MAX_STEPS):
            x, y = state
            if np.random.rand() < EPSILON:
                action = np.random.choice(list(actions.keys()))
            else:
                action = np.argmax(Q[x, y])
            dx, dy = actions[action]
            next_state = (x + dx, y + dy)

            # 벽 또는 맵 밖인지 확인
            if not (0 <= next_state[0] < SIZE and 0 <= next_state[1] < SIZE and env[next_state] == 0):
                next_state = state
            
            # 보상 설정
            reward = -1
            if next_state == goal: reward = 100
            elif next_state == state: reward = -10
            
            # Q-값 업데이트
            Q[x, y, action] += ALPHA * (reward + GAMMA * np.max(Q[next_state]) - Q[x, y, action])
            
            # 목표 도달 시 에피소드 종료
            if next_state == goal:
                succeeded = True
                break
            state = next_state
        
        episode += 1
        recent_outcomes.append(1 if succeeded else 0)

        # 수렴 조건 확인
        if len(recent_outcomes) == CONVERGENCE_CRITERIA_WINDOW:
            success_rate = np.mean(list(recent_outcomes))
            if success_rate >= CONVERGENCE_THRESHOLD:
                print(f"맵 #{map_index}: {episode} 에피소드 만에 수렴 성공!")
                is_converged = True
                break
    
    if not is_converged:
        print(f"맵 #{map_index}: 최대 에피소드({MAX_EPISODES}) 내에 수렴 실패.")

    return is_converged, episode

# ------------------------------------------------------------------
# 메인 테스트 실행 부분
# ------------------------------------------------------------------
if __name__ == "__main__":
    
    NUM_TEST_MAPS = 5  # 테스트할 맵의 개수
    success_count = 0
    total_episodes_for_success = 0

    for i in range(NUM_TEST_MAPS):
        converged, episodes_taken = train_on_single_map(i + 1)
        if converged:
            success_count += 1
            total_episodes_for_success += episodes_taken

    # --- 최종 결과 요약 ---
    print("\n\n<<<<< 최종 테스트 결과 요약 >>>>>")
    print(f"총 테스트 맵 수: {NUM_TEST_MAPS}개")
    print(f"성공적으로 수렴한 맵 수: {success_count}개")
    
    if NUM_TEST_MAPS > 0:
      success_rate = (success_count / NUM_TEST_MAPS) * 100
      print(f"알고리즘 성공률: {success_rate:.1f}%")

    if success_count > 0:
        avg_episodes = total_episodes_for_success / success_count
        print(f"성공한 경우, 평균 수렴 에피소드 수: {avg_episodes:.0f}회")