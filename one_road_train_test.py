import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from collections import deque

# ------------------------------------------------------------------
# 기본 함수 및 설정 (이전 코드와 동일)
# ------------------------------------------------------------------
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
SIZE = 50
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
MAX_STEPS = 2000
MAX_EPISODES = 10000
CONVERGENCE_CRITERIA_WINDOW = 100
CONVERGENCE_THRESHOLD = 0.98

# ------------------------------------------------------------------
# 파트 1: 하나의 맵에서 모델(Q-테이블) 훈련 (변경 없음)
# ------------------------------------------------------------------
def train_master_q_table():
    """하나의 훈련용 맵에서 Q-테이블을 학습시키고 그 결과를 반환합니다."""
    
    training_env, training_start, training_goal = generate_map(size=SIZE)
    Q = np.zeros((SIZE, SIZE, 4))
    
    print("===== 마스터 Q-테이블 훈련 시작 =====")
    print(f"훈련용 맵 생성 완료. 시작점: {training_start}, 목표점: {training_goal}")

    episode = 0
    recent_outcomes = deque(maxlen=CONVERGENCE_CRITERIA_WINDOW)
    
    while episode < MAX_EPISODES:
        state = training_start
        succeeded = False
        for _ in range(MAX_STEPS):
            x, y = state
            if np.random.rand() < EPSILON:
                action = np.random.choice(list(actions.keys()))
            else:
                action = np.argmax(Q[x, y])
            dx, dy = actions[action]
            next_state = (x + dx, y + dy)

            if not (0 <= next_state[0] < SIZE and 0 <= next_state[1] < SIZE and training_env[next_state] == 0):
                next_state = state
            
            reward = -1
            if next_state == training_goal: reward = 100
            elif next_state == state: reward = -10
            
            Q[x, y, action] += ALPHA * (reward + GAMMA * np.max(Q[next_state]) - Q[x, y, action])
            
            if next_state == training_goal:
                succeeded = True
                break
            state = next_state
        
        episode += 1
        recent_outcomes.append(1 if succeeded else 0)

        if len(recent_outcomes) == CONVERGENCE_CRITERIA_WINDOW:
            success_rate = np.mean(list(recent_outcomes))
            if episode % 500 == 0:
                 print(f"훈련 진행... (Episode {episode}, 최근 성공률: {success_rate:.2f})")
            if success_rate >= CONVERGENCE_THRESHOLD:
                print(f"훈련 성공! {episode} 에피소드 만에 Q-테이블이 수렴했습니다.")
                return Q
    
    print("훈련 실패: 최대 에피소드 내에 모델이 수렴하지 못했습니다.")
    return None

# ======================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
#                  파트 2: 학습된 모델로 새로운 맵 테스트 (수정된 부분)
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ======================================================================
def test_on_new_map(test_env, start, goal, master_Q):
    """학습된 Q-테이블을 사용하여 새로운 맵에서 경로를 찾는지 테스트합니다."""
    state = start
    path_taken = [start]
    succeeded = False

    for _ in range(MAX_STEPS):
        # 학습 없이 오직 Q-테이블의 가장 좋은 값만 따라감 (epsilon=0)
        action = np.argmax(master_Q[state])
        dx, dy = actions[action]
        potential_next_state = (state[0] + dx, state[1] + dy)

        # ★★★★★ 핵심 수정: 벽에 부딪히면 제자리에 머물도록 변경 ★★★★★
        # 다음 상태가 유효한지(맵 안이고, 벽이 아닌지) 확인
        if (0 <= potential_next_state[0] < SIZE and 
            0 <= potential_next_state[1] < SIZE and 
            test_env[potential_next_state] == 0):
            # 유효한 경우에만 상태 업데이트
            state = potential_next_state
        # else: # 유효하지 않은 경우(벽 충돌), state는 그대로 유지됨 (제자리에 머무름)

        path_taken.append(state)

        # 목표 도달 시 성공 처리 후 루프 탈출
        if state == goal:
            print("테스트 성공: 목표 지점에 도달했습니다!")
            succeeded = True
            break
            
    # for 루프가 끝난 후에도 성공하지 못했다면 실패 처리
    if not succeeded:
        print("테스트 실패: 최대 스텝 내에 목표에 도달하지 못했습니다.")
        
    return succeeded, path_taken

# ------------------------------------------------------------------
# 메인 실행 부분 (변경 없음)
# ------------------------------------------------------------------
if __name__ == "__main__":
    
    trained_Q = train_master_q_table()
    
    if trained_Q is not None:
        NUM_TEST_MAPS = 4
        test_success_count = 0
        
        print(f"\n===== 훈련된 모델로 {NUM_TEST_MAPS}개의 새로운 맵 테스트 시작 =====")

        for i in range(NUM_TEST_MAPS):
            print(f"\n--- 테스트 맵 #{i+1} ---")
            test_env, test_start, test_goal = generate_map(size=SIZE)
            
            succeeded, path = test_on_new_map(test_env, test_start, test_goal, trained_Q)
            if succeeded:
                test_success_count += 1
            
            plt.figure(figsize=(6, 6))
            final_visual = np.stack([test_env.copy().astype(float)] * 3, axis=-1)
            for pos in path:
                if pos != test_start and pos != test_goal:
                    final_visual[pos] = [1, 1, 0]
            final_visual[test_start] = [0, 1, 0]
            final_visual[test_goal] = [0, 0, 1]
            plt.imshow(final_visual)
            plt.title(f"Test Map #{i+1} - Result: {'Success' if succeeded else 'Failure'}")
            plt.axis('off')
            plt.show()

        print("\n\n<<<<< 최종 테스트 결과 요약 >>>>>")
        print("훈련된 하나의 모델을 새로운 맵에 적용한 결과입니다.")
        print(f"총 테스트 맵 수: {NUM_TEST_MAPS}개")
        print(f"성공적으로 길을 찾은 맵 수: {test_success_count}개")
        success_rate = (test_success_count / NUM_TEST_MAPS) * 100
        print(f"성공률: {success_rate:.1f}%")