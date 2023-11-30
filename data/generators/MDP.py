from collections import defaultdict
from data.generators.Abstract import *


class State:
    def __init__(self, X: List[Value], reward: float):
        self.X = X
        self.R = reward

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        # TODO compare using numpy
        for i in range(len(self.X)):
            if self.X[i] != other.X[i]:
                return False
        return True

    def __str__(self):
        return f"State([{', '.join(map(str, self.X))}], {self.R})"


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

    def add_states(self, states: List[State]) -> bool:
        res = True
        for state in states:
            res = self.add_state(state) & res
        return res

    def add_transition(self, from_state: State, action: Action, to_state: State, p: float):
        if from_state not in self.T:
            self.T[from_state] = dict()
        if action not in self.T[from_state]:
            self.T[from_state][action] = dict()
        self.T[from_state][action][to_state] = p
        assert sum(self.T[from_state][action].values()) <= 1, f"Sum of probabilities of T({from_state}, {action}) is greater than 1."

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
        return np.random.choice(list(possible_outcomes.keys()), p=list(possible_outcomes.values()))

    def perform_actions(self, state: State, actions: List[Action]) -> Tuple[State, float]:
        reward = state.R
        for action in actions:
            state = self.perform_action(state, action)
            reward += state.R
        return state, reward
