import math

from typing import *
from data.builders import *
from data.generators import *


class MDPBuilder:

    def __init__(self, mdp_name: str, state_size: int, action_size: int):
        self.mdp = MDP(mdp_name, state_size, action_size)
        self.states = dict()
        self.actions = dict()
        self.action_list = list()

    def get_state(self, name: str) -> State:
        return self.states[name]

    def add_action(self, name: str, a: List[Value]) -> Self:
        assert name not in self.actions, f"Action {name} already exists!"
        self.actions[name] = Action(a)
        self.action_list.append(self.actions[name])
        return self

    def add_constant_discrete_state(self, name: str, values: List[Any], reward: float, terminal: bool=False) -> Self:
        assert name not in self.states, f"State {name} already exists!"
        state = State([Constant(values[i], is_continuous=False) for i in range(self.mdp.state_size)], reward, is_terminal=terminal)
        self.states[name] = state
        self.mdp.add_state(state)
        return self

    def add_constant_uniform_continuous_state(self, name: str, ranges: List[Tuple[float, float]],
                                              reward: float) -> Self:
        assert name not in self.states, f"State {name} already exists!"
        state = State([Constant(None,
                                          range=ranges[i],
                                          is_continuous=True,
                                          generator=lambda: ranges[i][0] + (
                                                      ranges[i][1] - ranges[i][0]) * np.random.rand()) for i in
                                 range(self.mdp.state_size)],
                      reward)
        self.states[name] = state
        self.mdp.add_state(state)
        return self

    def add_constant_normal_continuous_state(self, name: str, means: List[float], stds: List[float],
                                             reward: float) -> Self:
        assert name not in self.states, f"State {name} already exists!"
        state = State([Constant(None,
                                          range=(-math.inf, math.inf),
                                          is_continuous=True,
                                          generator=lambda: means[i] + stds[i] * np.random.randn()) for i
                                 in range(self.mdp.state_size)],
                      reward)
        self.states[name] = state
        self.mdp.add_state(state)
        return self

    def add_constant_mixed_uniform_state(self, name: str, ranges: List[Tuple[float, float]],
                                         reward: float) -> Self:
        assert name not in self.states, f"State {name} already exists!"
        constants = []
        for i in range(self.mdp.state_size):
            if len(ranges[i]) == 2:
                constants.append(Constant(None,
                                          range=ranges[i],
                                          is_continuous=True,
                                          generator=lambda: ranges[i][0] + (
                                                      ranges[i][1] - ranges[i][0]) * np.random.rand()))
            else:
                constants.append(Constant(ranges[i][0], is_continuous=False))
        state = State(constants, reward)
        self.states[name] = state
        self.mdp.add_state(state)
        return self

    def connect_states(self, from_state: str, to_state: str, action: str,
                       p_lambda: Callable[[List[Value]], float]) -> Self:
        self.mdp.add_transition(self.states[from_state], self.actions[action], self.states[to_state], p_lambda)
        return self

    def connect_from_all(self, to_state: str, action: str | List[str],
                         p_lambda: Callable[[List[Value]], float] | List[Callable[[List[Value]], float]],
                         connect_to_self: bool = True) -> Self:
        for i, state in enumerate(self.states):
            if state == to_state and not connect_to_self:
                continue
            if type(action) is list:
                curr_action = action[i]
            else:
                curr_action = action
            if type(p_lambda) is list:
                curr_p = p_lambda[i]
            else:
                curr_p = p_lambda
            self.connect_states(state, to_state, curr_action, curr_p)
        return self

    def connect_to_all(self, from_state: str, action: str | List[str],
                       p_lambda: Callable[[List[Value]], float] | List[Callable[[List[Value]], float]],
                       connect_to_self: bool = True) -> Self:
        for i, state in enumerate(self.states):
            if state == from_state and not connect_to_self:
                continue
            if type(action) is str:
                curr_action = action[i]
            else:
                curr_action = action
            if type(p_lambda) is list:
                curr_p = p_lambda[i]
            else:
                curr_p = p_lambda
            self.connect_states(from_state, state, curr_action, curr_p)
        return self

    def add_custom_state(self, name: str, state: State) -> Self:
        assert name not in self.states, f"State {name} already exists!"
        self.states[name] = state
        self.mdp.add_state(state)
        return self

    def connect_self_with_actions(self, state: str, actions: List[str], p_lambdas: List[Callable]):
        for index, action in enumerate(actions):
            self.connect_states(state, state, action, p_lambdas[index])
        return self

    def build(self) -> MDP:
        return self.mdp
