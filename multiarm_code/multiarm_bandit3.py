import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class MultiArmedBandit:
    def __init__(self, k):
        self.k = k
        self.probs = np.random.rand(k)

    def pull(self, arm):
        return np.random.rand() < self.probs[arm]

    def optimal_reward(self):
        return np.max(self.probs)

    def optimal_arm(self):
        return np.argmax(self.probs)

def ucb_policy(bandit, time_steps, c):
    k = bandit.k
    counts = np.ones(k)
    rewards = np.zeros(k)

    total_reward = 0
    regret = []
    reward_log = []
    arm_log = []

    for i in range(k):
        reward = bandit.pull(i)
        rewards[i] += reward
        total_reward += reward
        regret.append(bandit.optimal_reward() * (i + 1) - total_reward)
        reward_log.append(reward)
        arm_log.append(i)

    for t in range(k + 1, time_steps + 1):
        avg_rewards = rewards / counts
        confidence_bounds = c * np.sqrt(np.log(t) / counts)
        ucb_values = avg_rewards + confidence_bounds
        arm = np.argmax(ucb_values)

        reward = bandit.pull(arm)
        counts[arm] += 1
        rewards[arm] += reward
        total_reward += reward

        reward_log.append(reward)
        arm_log.append(arm)

        optimal = bandit.optimal_reward() * t
        regret.append(optimal - total_reward)

    return regret, reward_log, arm_log, bandit.optimal_arm()

def run_experiment(k=10, time_steps=10000, trials=100, c_values=[0, 0.1, 0.5, 1, np.sqrt(2), 2, 5]):
    avg_regrets = {c: np.zeros(time_steps) for c in c_values}
    avg_rewards = {c: np.zeros(time_steps) for c in c_values}
    arm_selection_stats = {c: {'optimal': 0, 'total': 0} for c in c_values}

    for _ in tqdm(range(trials), desc="Running trials"):
        bandit = MultiArmedBandit(k)
        for c in c_values:
            regret, rewards, arm_log, best_arm = ucb_policy(bandit, time_steps, c)
            avg_regrets[c] += regret
            avg_rewards[c] += rewards
            arm_selection_stats[c]['optimal'] += sum(1 for a in arm_log if a == best_arm)
            arm_selection_stats[c]['total'] += len(arm_log)

    for c in c_values:
        avg_regrets[c] /= trials
        avg_rewards[c] /= trials

    return avg_regrets, avg_rewards, arm_selection_stats


def plot_regrets(avg_regrets):
    plt.figure(figsize=(10, 6))
    for c, regret in avg_regrets.items():
        plt.plot(regret, label=f'c = {c}')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Regret')
    plt.title('UCB Policy: Regret vs Time for Different c Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_avg_rewards(avg_rewards):
    plt.figure(figsize=(10, 6))
    for c, rewards in avg_rewards.items():
        plt.plot(rewards, label=f'c = {c}')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Reward')
    plt.title('UCB Policy: Average Reward per Time Step')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_arm_selection_stats(stats):
    labels = [str(c) for c in stats.keys()]
    optimal = [v['optimal'] / v['total'] for v in stats.values()]
    suboptimal = [1 - val for val in optimal]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, optimal, width, label='Optimal Arm')
    plt.bar(x + width/2, suboptimal, width, label='Suboptimal Arms')
    plt.xticks(x, labels)
    plt.xlabel('Exploration Parameter c')
    plt.ylabel('Proportion of Arm Selections')
    plt.title('Optimal vs Suboptimal Arm Selections')
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    regrets, rewards, stats = run_experiment()
    plot_regrets(regrets)
    plot_avg_rewards(rewards)
    plot_arm_selection_stats(stats)
