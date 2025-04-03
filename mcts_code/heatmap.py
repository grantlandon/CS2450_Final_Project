import numpy as np
import matplotlib.pyplot as plt
from mcts import MCTSAgent
from game_runner import run_many
import tqdm

c_values = [0, 0.5, 1, np.sqrt(2), 2]
agents = [MCTSAgent(c=c) for c in c_values]
agent_names = [f"MCTS(c={c})" for c in c_values]

num_agents = len(agents)
results_matrix = np.zeros((num_agents, num_agents), dtype=int)

for i in tqdm.tqdm(range(num_agents)):
    for j in range(i + 1, num_agents):
        print(f"\n=== Match: {agent_names[i]} vs {agent_names[j]} ===")
        score_i, score_j = run_many(agents[i], agents[j], num_games=50, verbose=True, size=5)
        results_matrix[i][j] = score_i
        results_matrix[j][i] = score_j

print("\n=== Tournament Results Matrix ===")
print(f"{'':20}", end="")
for name in agent_names:
    print(f"{name:>15}", end="")
print()
for i in range(num_agents):
    print(f"{agent_names[i]:20}", end="")
    for j in range(num_agents):
        if i == j:
            print(f"{'--':>15}", end="")
        else:
            print(f"{results_matrix[i][j]:>15}", end="")
    print()

plt.figure(figsize=(10, 8))
plt.imshow(results_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Agent i score vs Agent j')
plt.xticks(ticks=np.arange(num_agents), labels=agent_names, rotation=45, ha='right')
plt.yticks(ticks=np.arange(num_agents), labels=agent_names)
plt.title("MCTS Tournament Results Heatmap")
plt.xlabel("Opponent Agent (j)")
plt.ylabel("Evaluated Agent (i)")

for i in range(num_agents):
    for j in range(num_agents):
        if i != j:
            plt.text(j, i, str(results_matrix[i, j]),
                     ha='center', va='center', color='black', fontsize=10)

plt.tight_layout()
plt.show()
