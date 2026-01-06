"""
Multi-Armed Bandit (MAB) 강화학습 실습 코드

이 스크립트는 강화학습의 기초인 Multi-Armed Bandit 문제를 구현합니다.
여러 슬롯머신(Arm) 중에서 최적의 선택을 학습하는 과정을 시뮬레이션합니다.

구현된 알고리즘:
1. Epsilon-Greedy: 일정 확률로 탐험(exploration)과 활용(exploitation)을 수행
2. UCB (Upper Confidence Bound): 불확실성을 고려한 선택
3. Greedy: 항상 현재 최선으로 보이는 선택만 수행

주요 기능:
- 여러 알고리즘 성능 비교
- 시각화를 통한 학습 과정 분석
- 평균 보상 및 최적 행동 선택 비율 추적
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class Bandit:
    """
    Multi-Armed Bandit 환경
    
    각 arm(슬롯머신)은 고유한 평균 보상값을 가지며,
    선택 시 정규분포를 따르는 보상을 반환합니다.
    """
    
    def __init__(self, n_arms: int = 10):
        """
        Args:
            n_arms: 슬롯머신(팔)의 개수
        """
        self.n_arms = n_arms
        # 각 arm의 실제 평균 보상값 (정규분포에서 샘플링)
        self.true_rewards = np.random.randn(n_arms)
        self.best_arm = np.argmax(self.true_rewards)
        
    def pull(self, arm: int) -> float:
        """
        특정 arm을 선택하고 보상을 받습니다.
        
        Args:
            arm: 선택할 arm의 인덱스
            
        Returns:
            보상값 (실제 평균 + 노이즈)
        """
        reward = self.true_rewards[arm] + np.random.randn()
        return reward
    
    def get_optimal_reward(self) -> float:
        """최적 arm의 평균 보상값 반환"""
        return self.true_rewards[self.best_arm]


class EpsilonGreedy:
    """
    Epsilon-Greedy 알고리즘
    
    epsilon 확률로 무작위 탐험을 수행하고,
    1-epsilon 확률로 현재까지 최고 평균 보상의 arm을 선택합니다.
    """
    
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        """
        Args:
            n_arms: arm의 개수
            epsilon: 탐험 확률 (0~1)
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros(n_arms)  # 각 arm의 추정 가치
        self.action_counts = np.zeros(n_arms)  # 각 arm의 선택 횟수
        
    def select_arm(self) -> int:
        """arm 선택"""
        if np.random.random() < self.epsilon:
            # 탐험: 무작위 선택
            return np.random.randint(self.n_arms)
        else:
            # 활용: 최고 가치의 arm 선택
            return np.argmax(self.q_values)
    
    def update(self, arm: int, reward: float):
        """
        Q-value 업데이트
        
        증분 평균 공식 사용:
        Q_n+1 = Q_n + (1/n) * (R_n - Q_n)
        """
        self.action_counts[arm] += 1
        n = self.action_counts[arm]
        self.q_values[arm] += (reward - self.q_values[arm]) / n


class UCB:
    """
    Upper Confidence Bound (UCB) 알고리즘
    
    불확실성이 높은 arm에 보너스를 부여하여
    탐험과 활용의 균형을 자동으로 조절합니다.
    """
    
    def __init__(self, n_arms: int, c: float = 2.0):
        """
        Args:
            n_arms: arm의 개수
            c: 탐험 정도를 조절하는 파라미터
        """
        self.n_arms = n_arms
        self.c = c
        self.q_values = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms)
        self.total_counts = 0
        
    def select_arm(self) -> int:
        """UCB 기준으로 arm 선택"""
        # 모든 arm을 최소 1번씩 시도
        if self.total_counts < self.n_arms:
            return self.total_counts
        
        # UCB 값 계산
        ucb_values = self.q_values + self.c * np.sqrt(
            np.log(self.total_counts) / (self.action_counts + 1e-5)
        )
        return np.argmax(ucb_values)
    
    def update(self, arm: int, reward: float):
        """Q-value 및 카운트 업데이트"""
        self.action_counts[arm] += 1
        self.total_counts += 1
        n = self.action_counts[arm]
        self.q_values[arm] += (reward - self.q_values[arm]) / n


class GreedyAgent:
    """
    순수 Greedy 알고리즘
    
    탐험 없이 항상 현재까지 최고 평균 보상의 arm만 선택합니다.
    """
    
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.q_values = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms)
        
    def select_arm(self) -> int:
        """최고 Q-value의 arm 선택"""
        return np.argmax(self.q_values)
    
    def update(self, arm: int, reward: float):
        """Q-value 업데이트"""
        self.action_counts[arm] += 1
        n = self.action_counts[arm]
        self.q_values[arm] += (reward - self.q_values[arm]) / n


def run_experiment(
    agent_class,
    agent_params: dict,
    n_arms: int = 10,
    n_steps: int = 1000,
    n_runs: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    실험 실행 및 결과 수집
    
    Args:
        agent_class: 사용할 에이전트 클래스
        agent_params: 에이전트 초기화 파라미터
        n_arms: arm 개수
        n_steps: 각 실행당 스텝 수
        n_runs: 반복 실행 횟수
        
    Returns:
        평균 보상 배열, 최적 행동 선택 비율 배열
    """
    all_rewards = np.zeros((n_runs, n_steps))
    all_optimal_actions = np.zeros((n_runs, n_steps))
    
    for run in range(n_runs):
        bandit = Bandit(n_arms)
        agent = agent_class(**agent_params)
        
        for step in range(n_steps):
            arm = agent.select_arm()
            reward = bandit.pull(arm)
            agent.update(arm, reward)
            
            all_rewards[run, step] = reward
            all_optimal_actions[run, step] = (arm == bandit.best_arm)
    
    avg_rewards = np.mean(all_rewards, axis=0)
    optimal_action_rate = np.mean(all_optimal_actions, axis=0)
    
    return avg_rewards, optimal_action_rate


def plot_results(results: dict, n_steps: int):
    """
    실험 결과 시각화
    
    Args:
        results: {알고리즘명: (평균보상, 최적행동비율)} 딕셔너리
        n_steps: 스텝 수
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 평균 보상 그래프
    for name, (avg_rewards, _) in results.items():
        axes[0].plot(avg_rewards, label=name, linewidth=2)
    
    axes[0].set_xlabel('스텝', fontsize=12)
    axes[0].set_ylabel('평균 보상', fontsize=12)
    axes[0].set_title('알고리즘별 평균 보상 비교', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 최적 행동 선택 비율 그래프
    for name, (_, optimal_rate) in results.items():
        axes[1].plot(optimal_rate * 100, label=name, linewidth=2)
    
    axes[1].set_xlabel('스텝', fontsize=12)
    axes[1].set_ylabel('최적 행동 선택 비율 (%)', fontsize=12)
    axes[1].set_title('알고리즘별 최적 행동 선택 비율', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bandit_results.png', dpi=300, bbox_inches='tight')
    print("\n결과 그래프가 'bandit_results.png'로 저장되었습니다.")
    plt.show()


def print_final_statistics(results: dict):
    """최종 통계 출력"""
    print("\n" + "="*60)
    print("최종 성능 통계 (마지막 100 스텝 평균)")
    print("="*60)
    
    for name, (avg_rewards, optimal_rate) in results.items():
        final_reward = np.mean(avg_rewards[-100:])
        final_optimal = np.mean(optimal_rate[-100:]) * 100
        
        print(f"\n{name}:")
        print(f"  - 평균 보상: {final_reward:.4f}")
        print(f"  - 최적 행동 선택 비율: {final_optimal:.2f}%")


def main():
    """메인 실행 함수"""
    print("="*60)
    print("Multi-Armed Bandit 강화학습 실습")
    print("="*60)
    
    # 실험 설정
    n_arms = 10
    n_steps = 1000
    n_runs = 100
    
    print(f"\n실험 설정:")
    print(f"  - Arm 개수: {n_arms}")
    print(f"  - 스텝 수: {n_steps}")
    print(f"  - 반복 실행: {n_runs}회")
    
    # 각 알고리즘 실험
    algorithms = [
        ("Epsilon-Greedy (ε=0.1)", EpsilonGreedy, {"n_arms": n_arms, "epsilon": 0.1}),
        ("Epsilon-Greedy (ε=0.01)", EpsilonGreedy, {"n_arms": n_arms, "epsilon": 0.01}),
        ("UCB (c=2.0)", UCB, {"n_arms": n_arms, "c": 2.0}),
        ("Greedy", GreedyAgent, {"n_arms": n_arms}),
    ]
    
    results = {}
    
    print("\n실험 진행 중...")
    for name, agent_class, params in algorithms:
        print(f"  - {name} 실행 중...")
        avg_rewards, optimal_rate = run_experiment(
            agent_class, params, n_arms, n_steps, n_runs
        )
        results[name] = (avg_rewards, optimal_rate)
    
    print("\n실험 완료!")
    
    # 결과 시각화
    plot_results(results, n_steps)
    
    # 최종 통계 출력
    print_final_statistics(results)
    
    print("\n" + "="*60)
    print("주요 관찰 사항:")
    print("="*60)
    print("1. Epsilon-Greedy: 탐험 비율(ε)에 따라 성능이 달라집니다.")
    print("2. UCB: 초기에는 탐험을 많이 하다가 점차 활용으로 전환됩니다.")
    print("3. Greedy: 초기에 잘못된 선택에 갇힐 수 있습니다(Local Optimum).")
    print("="*60)


if __name__ == "__main__":
    main()

