import numpy as np
import matplotlib.pyplot as plt
import os
from threshhold_mcts import DecayingMCTSAgent
from mcts import MCTSAgent
from game_runner import run_many
import tqdm

os.makedirs("mcts_plots", exist_ok=True)

fixed_c_values = [0, 0.5, 1.0, np.sqrt(2), 2.0]
fixed_labels = [r"$c=0$", r"$c=0.50$", r"$c=1.00$", r"$c=1.41$", r"$c=2.00$"]

def run_dynamic_vs_fixed(board_size, num_games=10):
    results = []

    dynamic_agent = DecayingMCTSAgent(early_c=2.0, late_c=0.0)

    for c in tqdm.tqdm(fixed_c_values):
        fixed_agent = MCTSAgent(c=c)
        print(f"\n=== Match: Dynamic Agent vs Fixed Agent (c={c}) on {board_size}x{board_size} board ===")
        dynamic_score, fixed_score = run_many(dynamic_agent, fixed_agent, num_games=num_games, verbose=False, size=board_size)
        win_differential = dynamic_score - fixed_score
        results.append(win_differential)

    return np.array(results)

def plot_dynamic_vs_fixed(differentials, filename, title):
    fig, ax = plt.subplots(figsize=(10, 2))

    data = differentials.reshape(1, -1)

    cax = ax.imshow(data, cmap="RdBu", vmin=-50, vmax=50)

    ax.set_xticks(np.arange(len(fixed_labels)))
    ax.set_yticks([0])
    ax.set_xticklabels(fixed_labels)
    ax.set_yticklabels(["Dynamic c (2→0)"])

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(fixed_labels)):
        ax.text(i, 0, f"{differentials[i]}", ha="center", va="center", color="black")

    ax.set_title(title)

    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Win Differential (Dynamic - Fixed)")

    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    differentials = run_dynamic_vs_fixed(board_size=5, num_games=50)
    plot_dynamic_vs_fixed(
        differentials, 
        "mcts_plots/5x5_dynamic.png",
        "5x5 Go: Dynamic Exploration Agent vs Fixed c Agents"
    )
    differentials = run_dynamic_vs_fixed(board_size=19, num_games=50)
    plot_dynamic_vs_fixed(
        differentials, 
        "mcts_plots/19x19dynamic.png",
        "19x19 Go: Dynamic Exploration Agent vs Fixed c Agents"
    )

if __name__ == "__main__":
    main()
