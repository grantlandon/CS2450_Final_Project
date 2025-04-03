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

def ucb_policy(bandit, time_steps, c):
    k = bandit.k
    counts = np.ones(k)  # pull each arm once
    rewards = np.zeros(k)

    # Initial pulls
    total_reward = 0
    regret = []

    for i in range(k):
        reward = bandit.pull(i)
        rewards[i] += reward
        total_reward += reward
        regret.append(bandit.optimal_reward() * (i + 1) - total_reward)

    for t in range(k + 1, time_steps + 1):
        avg_rewards = rewards / counts
        confidence_bounds = c * np.sqrt(np.log(t) / counts)
        ucb_values = avg_rewards + confidence_bounds
        arm = np.argmax(ucb_values)

        reward = bandit.pull(arm)
        counts[arm] += 1
        rewards[arm] += reward
        total_reward += reward

        optimal = bandit.optimal_reward() * t
        regret.append(optimal - total_reward)

    return regret


def run_experiment(k=10, time_steps=10000, trials=100, c_values=[0, 0.1, 0.5, 1, np.sqrt(2), 2, 5]):
    avg_regrets = {c: np.zeros(time_steps) for c in c_values}

    for _ in tqdm(range(trials), desc="Running trials"):
        bandit = MultiArmedBandit(k)
        for c in c_values:
            regret = ucb_policy(bandit, time_steps, c)
            avg_regrets[c] += regret

    for c in c_values:
        avg_regrets[c] /= trials

    return avg_regrets

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

if __name__ == "__main__":
    regrets = run_experiment()
    plot_regrets(regrets)
