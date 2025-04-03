import numpy as np
import matplotlib.pyplot as plt

class UCBBandit:
    def __init__(self, k, c, true_means, steps):
        self.k = k
        self.c = c
        self.true_means = true_means
        self.steps = steps
        self.counts = np.zeros(k)
        self.values = np.zeros(k)
        self.total_regret = np.zeros(steps)
        self.best_mean = np.max(true_means)

    def select_action(self, t):
        ucb_values = np.zeros(self.k)
        for i in range(self.k):
            if self.counts[i] == 0:
                return i
            bonus = self.c * np.sqrt(np.log(t + 1) / self.counts[i])
            ucb_values[i] = self.values[i] + bonus
        return np.argmax(ucb_values)

    def update(self, action, reward):
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]

    def run(self):
        regret = np.zeros(self.steps)
        for t in range(self.steps):
            action = self.select_action(t)
            reward = np.random.normal(self.true_means[action], 1)
            self.update(action, reward)
            regret[t] = self.best_mean - self.true_means[action]
        return np.cumsum(regret)

k = 10
steps = 50000
runs = 100
c_values = [0, 0.1, 0.5, 1.0, np.sqrt(2), 2.0, 5.0]

avg_regrets = {}

for c in c_values:
    all_regrets = []
    for _ in range(runs):
        true_means = np.random.normal(0, 1, k)
        bandit = UCBBandit(k=k, c=c, true_means=true_means, steps=steps)
        regret = bandit.run()
        all_regrets.append(regret)
    avg_regrets[c] = np.mean(all_regrets, axis=0)

plt.figure(figsize=(10, 6))
for c in c_values:
    plt.plot(avg_regrets[c], label=f"c = {c}")
plt.title("Empirical Regret of UCB with Different Exploration Constants")
plt.xlabel("Steps")
plt.ylabel("Cumulative Regret")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
