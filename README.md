# CS2450_Final_Project

# CS2450 Final Project: Exploration Constants in Monte Carlo Tree Search (MCTS)

This project investigates the role of the exploration constant \\( c \\) in the UCT algorithm and its impact on the performance of Monte Carlo Tree Search (MCTS) agents across different domains.

## Overview

- Theoretically, \\( c = \sqrt{2} \\) minimizes regret in classical multi-armed bandit problems under standard assumptions (stationary, independent rewards).
- In practice, when applied to MCTS, these assumptions break down (non-stationarity, delayed feedback, action correlation), making \\( c \\) a tunable hyperparameter.
- This project explores both the theoretical background and empirical performance of different exploration constants in:
  - **Multi-Armed Bandits**
  - **5x5 Go**
  - **19x19 Go**
- It also introduces a **dynamic exploration agent** where the exploration constant decays over the course of a game.

## Structure

- `multi_arm_bandit_experiments/`: Simulations comparing different \\( c \\) values in a bandit setting.
- `mcts_experiments/`: Round-robin tournaments between MCTS agents with different \\( c \\) values.
- `dynamic_exploration_agent/`: Implementation and evaluation of a dynamic \\( c \\) strategy.
- `plots/`: Resulting heatmaps, performance graphs, and comparisons.
- `final_report.pdf`: Full NeurIPS-style paper summarizing the findings.

## Key Findings

- In simple stochastic settings, \\( c = \sqrt{2} \\) performs optimally, matching theory.
- In small games like **5x5 Go**, **smaller** exploration constants (e.g., \\( c = 0.5 \\)) outperform \\( \sqrt{2} \\).
- In larger, complex games like **19x19 Go**, \\( c = \sqrt{2} \\) becomes closer to optimal.
- **Dynamic exploration** strategies that reduce \\( c \\) over time outperform fixed strategies in both settings.

## References
This project draws on foundational work from Auer et al. (2002), Sutton and Barto (2018), Silver et al. (2016, 2017), Kocsis and Szepesvári (2006), and others (see References section in the report).
