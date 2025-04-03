from go_search_problem import GoProblem, GoState, Action
from adversarial_search_problem import GameState
from heuristic_go_problems import *
import random
from abc import ABC, abstractmethod
import numpy as np
import time


class GameAgent():
    @abstractmethod
    def get_move(self, game_state: GameState, time_limit: float) -> Action:
        pass

class MCTSNode:
    def __init__(self, state, parent=None, children=None, action=None):
        self.state = state
        self.parent = parent
        if children is None:
            children = []
        self.children = children
        self.visits = 0
        self.value = 0
        self.action = action

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"Node: {self.state} \nVisits: {self.visits} \nValue: {self.value} \n Parent: {self.parent}"

class MCTSAgent(GameAgent):
    def __init__(self, c=np.sqrt(2)):
        self.c = c
        self.search_problem = GoProblem()

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
        best_ucb = -float('inf')
        best_child = None
        while not node.state.is_terminal_state():
            if len(node.state.legal_actions()) == 0:
                return node
            if len(node.children) < len(node.state.legal_actions()):
                return node
            best_ucb = -float('inf')
            best_child = None
            for child in node.children:
                UCB = child.value / child.visits + self.c * \
                    np.sqrt(np.log(node.visits) / child.visits)
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
        actions = actions - already_explored
        action = np.random.choice(list(actions))
        new_state = self.search_problem.transition(leaf.state, action)
        new_node = MCTSNode(new_state, parent=leaf, action=action)
        leaf.children.append(new_node)
        new_node.parent = leaf
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
    
    print("=== Match: MCTS(c=1.4142135623730951) vs MCTS(c=2) ===")
    agent1 = MCTSAgent(c=np.sqrt(2))
    agent2 = MCTSAgent(c=2)
    run_many(agent1, agent2, num_games=4, verbose=True, size=7)
    
    print("=== Match: MCTS(c=1.4142135623730951) vs MCTS(c=1) ===")
    agent1 = MCTSAgent(c=np.sqrt(2))
    agent2 = MCTSAgent(c=1)
    run_many(agent1, agent2, num_games=4, verbose=True, size=7)
    
    print("=== Match: MCTS(c=1.4142135623730951) vs MCTS(c=0.5) ===")
    agent1 = MCTSAgent(c=np.sqrt(2))
    agent2 = MCTSAgent(c=0.5)
    run_many(agent1, agent2, num_games=4, verbose=True, size=7)

if __name__ == "__main__":
    main()