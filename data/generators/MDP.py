from collections import defaultdict
from data.generators.Abstract import *


class State:
    def __init__(self, X: List[Value], reward: float):
        self.X = X
        self.R = reward


class Action:
    def __init__(self, a: List[Value]):
        self.a = a


class MDP:

    def __init__(self, name: str):
        self.name = name
        self.S: Set[State] = set()
        self.T: Dict[State, Dict[Action, Dict[State, float]]] = dict()

    def add_state(self, state: State) -> bool:
        if state in self.S:
            return False
        self.S.add(state)
        return True

    def add_transition(self, from_state: State, action: Action, to_state: State, p: float):
        if from_state not in self.T:
            self.T[from_state] = dict()
        if action not in self.T[from_state]:
            self.T[from_state][action] = dict()
        self.T[from_state][action][to_state] = p
        assert sum(self.T[from_state][action].items()) <= 1, f"Sum of probabilities of T({from_state}, {action}) is greater than 1."

    def get_actions(self, state: State) -> Dict[Action, Dict[State, float]]:
        if state not in self.T:
            return dict()
        return self.T[state]

    def perform_action(self, state: State, action: Action) -> State:
        if state not in self.T:
            raise RuntimeError(f"State {state} is a Terminal state.")
        if action not in self.T[state]:
            raise RuntimeError(f"Action {action} not possible from state {state}.")
        possible_outcomes = self.T[state][action]
        return np.random.choice(np.array(possible_outcomes.keys()), p=np.array(possible_outcomes.values()))

    def perform_actions(self, state: State, actions: List[Action]) -> State:
        pass
