import math
from collections import defaultdict

import numpy as np

from data.generators.Abstract import *
from torch import Tensor
import tkinter


class Action:
    def __init__(self, a: List[Value]):
        self.a = a

    def __str__(self):
        return f"Action({self.a})"

    def __repr__(self):
        return self.__str__()


class State:
    def __init__(self, X: List[Value], reward: float, is_terminal: bool = False, name: str="State"):
        self.X = X
        self.R = reward
        self.terminal = is_terminal
        self.name = name

    def __hash__(self):
        return id(self)

    def __call__(self, *args, **kwargs) -> Tensor:
        return Tensor([x() for x in self.X])

    def __eq__(self, other):
        return id(self) == id(other)

    def __str__(self):
        return f"State({self.name})"

    def __repr__(self):
        return self.__str__()


class MDP:

    def __init__(self, name: str, state_size: int, action_size: int):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.S: Set[State] = set()
        self.T: Dict[State, Dict[Action, Dict[State, Callable[List[Value], float]]]] = dict()

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

    def add_transition(self, from_state: State, action: Action, to_state: State, p: Callable[[List[Value]], float]):
        assert not from_state.terminal, f"State {from_state} is terminal, thus cannot have actions going out."
        assert len(action.a) == self.action_size, f"Actions in MDP {self.name} have to be of size {self.action_size}"
        if from_state not in self.T:
            self.T[from_state] = dict()
        if action not in self.T[from_state]:
            self.T[from_state][action] = dict()
        self.T[from_state][action][to_state] = p

    def get_actions(self, state: State) -> Dict[Action, Dict[State, Callable[[List[Value]], float]]]:
        if state not in self.T:
            return dict()
        return self.T[state]

    def perform_action(self, state: State, action: Action, x: List[Value]) -> Optional[State]:
        if state.terminal or action is None:
            return None
        if state not in self.T:
            raise RuntimeError(f"State {state} is a Terminal state.")
        if action not in self.T[state]:
            return state
        possible_outcomes = self.T[state][action]
        return np.random.choice(list(possible_outcomes.keys()), p=list(map(lambda f: f(x), possible_outcomes.values())))

    def perform_actions(self, state: State, actions: List[Action], x: List[Value]) -> List[Tuple[State, Action]]:
        history = [(state, None)]
        for action in actions:
            state = self.perform_action(state, action, x)
            if state is None:
                return history
            history.append((state, action))
        return history

    def visualize(self, repulsion_strength: float = 20, spring_stregth: float = 0.1, max_iter: int = 1000):
        width = 500
        height = 500
        radius = 15
        points = []
        index_to_state = dict()
        state_to_index = dict()
        num_of_rows = 5
        num_of_cols = len(self.S) // num_of_rows + 1
        for index, state in enumerate(self.S):
            row = index // num_of_rows
            col = index % num_of_cols
            point = np.array([(row / num_of_rows) * width, (col / num_of_cols) * height])
            index_to_state[index] = state
            state_to_index[state] = index
            points.append(point)
        c = 0
        spring_length = 100
        total_energy = math.inf
        minimum_energy = 1000
        while c < max_iter and minimum_energy < total_energy:
            total_energy = 0
            for index, point in enumerate(points):
                diff = np.zeros((2,))
                for j, other in enumerate(points):
                    if index == j:
                        continue
                    diff += repulsion_strength / (np.linalg.norm(point - other)**2) * (other - point) / np.linalg.norm(point - other)
                other_ids = set()
                if index_to_state[index] in self.T:
                    for other_state in map(lambda d: d.keys(), self.T[index_to_state[index]].values()):
                        for state in other_state:
                            if state_to_index[state] == index:
                                continue
                            other_ids.add(state_to_index[state])
                for j in other_ids:
                    other = points[j]
                    diff += spring_stregth * abs(np.linalg.norm(point - other) - spring_length) * (point - other) / np.linalg.norm(point - other)
                points[index] += diff
                total_energy += np.linalg.norm(diff)
            c += 1
        min_x = min([point[0] for point in points]) - 2 * radius
        max_x = max([point[0] for point in points]) + 2 * radius
        min_y = min([point[1] for point in points]) - 2 * radius
        max_y = max([point[1] for point in points]) + 2 * radius
        for index, point in enumerate(points):
            points[index] = np.array([(point[0] - min_x) / (max_x - min_x) * width, (point[1] - min_y) / (max_y - min_y) * height])
        root = tkinter.Tk()
        canvas = tkinter.Canvas(root, width=width, height=height)
        for index, point in enumerate(points):
            canvas.create_oval(point[0]-radius, point[1]-radius, point[0]+radius, point[1]+radius)
            canvas.create_text(point[0], point[1], text=index_to_state[index].name)
        canvas.pack()
        root.mainloop()

    def __str__(self):
        return f"MDP({self.name})"
