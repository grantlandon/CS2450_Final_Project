import numpy as np
import matplotlib.pyplot as plt
import os
from mcts import MCTSAgent
from game_runner import run_many
import tqdm

os.makedirs("mcts_plots", exist_ok=True)

c_values = [0, 0.5, 1, np.sqrt(2), 2]
agent_names = [f"MCTS(c={c})" for c in c_values]

def run_tournament(board_size, num_games=10):
    num_agents = len(c_values)
    results_matrix = np.zeros((num_agents, num_agents), dtype=int)

    agents = [MCTSAgent(c=c) for c in c_values]

    for i in tqdm.tqdm(range(num_agents)):
        for j in range(i + 1, num_agents):
            agent_i = agents[i]
            agent_j = agents[j]
            print(f"\n=== Match: {agent_names[i]} vs {agent_names[j]} on {board_size}x{board_size} board ===")
            agent_i_score, agent_j_score = run_many(agent_i, agent_j, num_games=num_games, verbose=False, size=board_size)
            results_matrix[i, j] = agent_i_score
            results_matrix[j, i] = agent_j_score

    return results_matrix

def plot_results(matrix, filename, title):
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(c_values)))
    ax.set_yticks(np.arange(len(c_values)))
    ax.set_xticklabels(c_values)
    ax.set_yticklabels(c_values)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(len(c_values)):
        for j in range(len(c_values)):
            text = ax.text(j, i, matrix[i, j],
                           ha="center", va="center", color="black")

    ax.set_title(title)
    fig.tight_layout()

    plt.savefig(filename)
    plt.close()

def main():
    results_5x5 = run_tournament(board_size=5, num_games=50)
    plot_results(results_5x5, "mcts_plots/5x5.png", "5x5 Board MCTS Match Results")

    results_19x19 = run_tournament(board_size=19, num_games=50)
    plot_results(results_19x19, "mcts_plots/19x19.png", "19x19 Board MCTS Match Results")

if __name__ == "__main__":
    main()
