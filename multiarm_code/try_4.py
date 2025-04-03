import numpy as np
import matplotlib.pyplot as plt

class UCB1:
    def __init__(self, n_arms, c):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.c = c
        self.total_counts = 0

    def select_arm(self):
        if self.total_counts < self.n_arms:
            return self.total_counts
        ucb_values = self.values + self.c * np.sqrt(np.log(self.total_counts) / (self.counts + 1e-9))
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] += (reward - value) / n

def run_simulation(true_means, c, time_horizon, n_trials):
    n_arms = len(true_means)
    regrets = np.zeros(time_horizon)

    for _ in range(n_trials):
        bandit = UCB1(n_arms, c)
        cumulative_regret = 0
        optimal_mean = np.max(true_means)

        for t in range(time_horizon):
            arm = bandit.select_arm()
            reward = np.random.normal(true_means[arm], 1.0)
            bandit.update(arm, reward)
            regret = optimal_mean - true_means[arm]
            cumulative_regret += regret
            regrets[t] += cumulative_regret

    return regrets / n_trials

# Parameters
true_means = [0.1, 0.5, 0.6, 0.7, 0.9]
time_horizon = 5000
n_trials = 100

# Add sqrt(2) to the range of c values
c_values = sorted(set(np.round(np.linspace(0, 3.0, 20), 4).tolist() + [np.sqrt(2)]))

# Run simulations
results = {}
for c in c_values:
    avg_regret = run_simulation(true_means, c, time_horizon, n_trials)
    results[c] = avg_regret[-1]
    print(f"c = {c:.4f}, Final Avg Regret = {avg_regret[-1]:.3f}")

# Plot
best_c = min(results, key=results.get)
plt.figure(figsize=(8, 5))
plt.plot(list(results.keys()), list(results.values()), marker='o')
plt.axvline(x=np.sqrt(2), linestyle='--', color='gray', label=r'$\sqrt{2}$')
plt.xlabel("Exploration parameter c")
plt.ylabel("Final average regret")
plt.title(f"Optimal c: {best_c:.4f} (Lowest Final Regret)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
