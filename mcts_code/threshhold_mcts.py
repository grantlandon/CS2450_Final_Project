from go_search_problem import GoProblem, GoState, Action
from adversarial_search_problem import GameState
from heuristic_go_problems import *
import random
from abc import ABC, abstractmethod
import numpy as np
import time
from mcts import MCTSNode, MCTSAgent


class GameAgent():
    @abstractmethod
    def get_move(self, game_state: GameState, time_limit: float) -> Action:
        pass


class DecayingMCTSAgent(GameAgent):
    def __init__(self, early_c=np.sqrt(2), late_c=0.5):
        """
        :param early_c: Exploration constant for early game
        :param late_c: Exploration constant for late game
        :param switch_threshold: Fraction of board filled before switching to late_c
        """
        self.early_c = early_c
        self.late_c = late_c
        self.search_problem = GoProblem()

    def get_c(self, state):
        board = state.get_board()
        empty_board = board[2]
        total_cells = empty_board.size
        empty_cells = np.sum(empty_board)
        filled_ratio = 1 - (empty_cells / total_cells)

        return (1 - filled_ratio) * self.early_c + filled_ratio * self.late_c

    def get_move(self, state, time_limit):
        tree = MCTSNode(state)
        start_time = time.time()
        counter = 0
        while time.time() - start_time < 1.0:
            leaf = self.select(tree)
            child = self.expand(leaf)
            result = self.simulate(child)
            self.backpropagate(child, result)
            counter += 1
        most_visits = 0
        best_action = None
        for child in tree.children:
            if child.visits > most_visits and child.action in state.legal_actions():
                most_visits = child.visits
                best_action = child.action
        return best_action

    def select(self, node):
        while not node.state.is_terminal_state():
            if len(node.state.legal_actions()) == 0:
                return node
            if len(node.children) < len(node.state.legal_actions()):
                return node
            best_ucb = -float('inf')
            best_child = None
            for child in node.children:
                c = self.get_c(node.state)
                UCB = child.value / child.visits + c * np.sqrt(np.log(node.visits) / child.visits)
                if UCB > best_ucb:
                    best_ucb = UCB
                    best_child = child
            node = best_child
        return node

    def expand(self, leaf):
        if leaf.state.is_terminal_state():
            return leaf
        actions = set(leaf.state.legal_actions())
        already_explored = set([child.action for child in leaf.children])
        unexplored_actions = actions - already_explored
        action = np.random.choice(list(unexplored_actions))
        new_state = self.search_problem.transition(leaf.state, action)
        new_node = MCTSNode(new_state, parent=leaf, action=action)
        leaf.children.append(new_node)
        return new_node

    def simulate(self, node):
        state = node.state
        while not state.is_terminal_state():
            actions = state.legal_actions()
            action = np.random.choice(actions)
            state = self.search_problem.transition(state, action)
        result = self.search_problem.evaluate_terminal(state)
        return result

    def backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            if node.state.player_to_move() == 1:
                if result > 0:
                    node.value += 1
            else:
                if result < 0:
                    node.value += 1
            node = node.parent


def main():
    from game_runner import run_many

    print("=== Match: Dynamic MCTS vs Fixed MCTS (c=2) ===")
    dynamic_agent = DecayingMCTSAgent(early_c=2, late_c=0.0)
    fixed_agent = MCTSAgent(c=1)
    run_many(dynamic_agent, fixed_agent, num_games=10, verbose=True, size=5)
    
    print("=== Match: Dynamic MCTS vs Fixed MCTS (c=1.41) ===")
    dynamic_agent = DecayingMCTSAgent(early_c=2.0, late_c=0.0)
    fixed_agent = MCTSAgent(c=np.sqrt(2))
    run_many(dynamic_agent, fixed_agent, num_games=50, verbose=True, size=5)
    
    print("=== Match: Dynamic MCTS vs Fixed MCTS (c=1) ===")
    dynamic_agent = DecayingMCTSAgent(early_c=2.0, late_c=0.00)
    fixed_agent = MCTSAgent(c=1)
    run_many(dynamic_agent, fixed_agent, num_games=50, verbose=True, size=5)
    
    print("=== Match: Dynamic MCTS vs Fixed MCTS (c=0.5) ===")
    dynamic_agent = DecayingMCTSAgent(early_c=2.0, late_c=0.0)
    fixed_agent = MCTSAgent(c=0.5)
    run_many(dynamic_agent, fixed_agent, num_games=50, verbose=True, size=5)
    
    print("=== Match: Dynamic MCTS vs Fixed MCTS (c=0) ===")
    dynamic_agent = DecayingMCTSAgent(early_c=2.0, late_c=0.0)
    fixed_agent = MCTSAgent(c=0.5)
    run_many(dynamic_agent, fixed_agent, num_games=50, verbose=True, size=5)


if __name__ == "__main__":
    main()
