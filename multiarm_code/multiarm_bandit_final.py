import numpy as np
import matplotlib.pyplot as plt

class UCB:
    def __init__(self, arm_count, c):
        self.arm_count = arm_count # number of arms
        self.counts = np.zeros(arm_count)  # number times each arm has been pulled
        self.values = np.zeros(arm_count)  # empirical average reward for each arm
        self.c = c  # exploration parameter 
        self.total_counts = 0  # total number of arm pulls

    def select_arm(self):
        if self.total_counts < self.arm_count:
            return self.total_counts
        ucb_values = self.values + self.c * np.sqrt(np.log(self.total_counts) / self.counts)
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] += (reward - value) / n

def run_simulation(c, pull_count, trial_count, arm_count=10):
    regrets = np.zeros(pull_count)
    optimal_arm_selection = 0
    total_arm_selection = np.zeros(arm_count)
    reward_curve = np.zeros(pull_count)

    for _ in range(trial_count):
        true_probs = np.random.uniform(0.2, 0.8, arm_count)
        optimal_arm = np.argmax(true_probs)
        bandit = UCB(arm_count, c)

        for t in range(pull_count):
            arm = bandit.select_arm()
            reward = np.random.binomial(1, true_probs[arm]) 
            bandit.update(arm, reward)
            regret = true_probs[optimal_arm] - true_probs[arm]
            regrets[t] += regret 
            total_arm_selection[arm] += 1
            reward_curve[t] += reward
            if arm == optimal_arm:
                optimal_arm_selection += 1

    percent_optimal = 100 * optimal_arm_selection / (pull_count * trial_count)
    avg_reward_per_timestep = reward_curve / trial_count
    return regrets / trial_count, percent_optimal, total_arm_selection / (pull_count * trial_count), avg_reward_per_timestep

# parameters
pull_count = 10000
trial_count = 100
arm_count = 100
c_values = sorted(np.linspace(0, 2, 9).tolist() + [np.sqrt(2)])

results = {}
optimal_selection_percents = {}
suboptimal_selection_distribution = {}
reward_curves = {}

for c in c_values:
    avg_regret, optimal_percent, arm_dist, reward_curve = run_simulation(c, pull_count, trial_count, arm_count)
    results[c] = avg_regret[-1]
    optimal_selection_percents[c] = optimal_percent
    suboptimal_selection_distribution[c] = arm_dist
    reward_curves[c] = reward_curve
    print(f"c = {c:.4f}, Final Avg Regret = {avg_regret[-1]:.3f}, Optimal Arm % = {optimal_percent:.2f}")

# final average regret plot
best_c = min(results, key=results.get)
plt.figure(figsize=(8, 5))
plt.plot(list(results.keys()), list(results.values()), marker='o')
plt.axvline(x=np.sqrt(2), linestyle='--', color='gray', label=r'$\sqrt{2}$')
plt.xlabel("Exploration parameter c")
plt.ylabel("Final average regret per timestep")
plt.title(f"Optimal c: {best_c:.4f} (Lowest Final Avg Regret)\n{arm_count} Random Bernoulli Arms")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# optimal arm selection percentages plot
plt.figure(figsize=(8, 5))
plt.bar([str(round(c, 2)) for c in c_values], [optimal_selection_percents[c] for c in c_values])
plt.xlabel("Exploration parameter c")
plt.ylabel("% of times optimal arm selected")
plt.title("Optimal Arm Selection Rate per c")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# smoothed average reward plot
def moving_average(x, w=100):
    return np.convolve(x, np.ones(w)/w, mode='valid')

plt.figure(figsize=(10, 6))
for c in c_values:
    smoothed = moving_average(reward_curves[c])
    plt.plot(smoothed, label=f"c={round(c,2)}")
plt.xlabel("Timestep")
plt.ylabel("Average Reward (smoothed)")
plt.title("Smoothed Average Reward per Timestep")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
