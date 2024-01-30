from collections import defaultdict
from typing import Dict

import pandas as pd
import sys

import pyomo.core
import torch

from models import *

from pyomo.environ import *

from pyomo.util.model_size import build_model_size_report


def create_feature_dict(state_features: List[str], action_features: List[str],
                        confounder_features: List[str], state_action_features: List[str],
                        next_state_features: List[str]):
    return {
        "state_features": state_features,
        "action_features": action_features,
        "confounder_features": confounder_features,
        "state_action_features": state_action_features,
        "next_state_features": next_state_features
    }


class Evaluator:

    def __init__(self, behavioral_policy_delta: float, transitional_delta: float):
        assert 0 < behavioral_policy_delta < 1 and 0 < transitional_delta < 1, "Delta must be less than 1 and greater than 0"
        self.delta1 = behavioral_policy_delta
        self.delta2 = transitional_delta

    def find_mdp(self, data: pd.DataFrame, transition_set: pd.DataFrame, feature_dict: Dict[str, List[str]], agent: Agent,
                 state_set: pd.DataFrame, action_set: pd.DataFrame, terminal: List[bool], maximize: bool = False) -> float:
        model = ConcreteModel(name="ValueFunctionModel")
        model.state = RangeSet(0, len(state_set)-1)
        model.action = RangeSet(0, len(action_set)-1)
        model.P = Var(model.state, model.action, model.state, bounds=(0, 1))
        model.V = Var(model.state, bounds=(-50, 50))
        gamma = 0.95

        indexer = dict()
        reward = defaultdict(int)
        for state in range(len(state_set)):
            for action in range(len(action_set)):
                for next_state in range(len(state_set)):
                    lookup = transition_set[
                        np.all(transition_set[feature_dict["state_features"]] == tuple(state_set.iloc[state].tolist()),
                               axis=1) &
                        np.all(
                            transition_set[feature_dict["action_features"]] == tuple(action_set.iloc[action].tolist()),
                            axis=1) &
                        np.all(transition_set[feature_dict["next_state_features"]] == tuple(
                            state_set.iloc[next_state].tolist()), axis=1)]
                    if len(lookup) > 0:
                        if state not in indexer:
                            indexer[state] = dict()
                        if action not in indexer[state]:
                            indexer[state][action] = dict()
                        indexer[state][action][next_state] = lookup.index[0]

        for next_state in range(len(state_set)):
            lookup = data[
                np.all(data[feature_dict["next_state_features"]] == tuple(state_set.iloc[next_state].tolist()),
                       axis=1)]
            reward[next_state] = lookup["R"].mean()

        agent_policy = []
        for _, state in state_set.iterrows():
            agent_policy.append(agent.policy(torch.Tensor(state)))
        agent_policy = np.array(agent_policy)

        # Constraint 1: Transitions are withing expected bounds
        def bound_constraint(model, state, action, next_state):
            if state not in indexer or action not in indexer[state] or next_state not in indexer[state][action] or terminal[state]:
                return model.P[state, action, next_state] == 0
            id = indexer[state][action][next_state]
            return transition_set.iloc[id]["t_lower"], model.P[state, action, next_state], transition_set.iloc[id][
                "t_upper"]

        model.c1 = Constraint(model.state, model.action, model.state, rule=bound_constraint)

        # Constraint 2: Transition probabilities sum up to 1
        def probability_sum_constraint(model, state, action):
            if state not in indexer or action not in indexer[state] or terminal[state]:
                return sum([model.P[state, action, j] for j in range(len(state_set))]) == 0
            return sum([model.P[state, action, j] for j in range(len(state_set))]) == 1

        model.c2 = Constraint(model.state, model.action, rule=probability_sum_constraint)

        # Constraint 3: Value function definition
        # V(s) = pi_e(.|s) * (R(s, a) + P(s'|s,a) * V(s'))
        def value_function_constraint(model, state):
            if terminal[state]:
                return model.V[state] == reward[state]
            return model.V[state] - sum(
                [agent_policy[state][action] * gamma * (reward[state] + sum([
                        model.P[state, action, next_state] * model.V[next_state]
                     for next_state in model.state
                     ]))
                 for action in model.action
                 ]) == 0

        model.c3 = Constraint(model.state, rule=value_function_constraint)

        # Objective: V(s_0)
        def objective_function(model):
            return model.V[0]

        model.OBJ = Objective(rule=objective_function,
                              sense=pyomo.core.minimize if not maximize else pyomo.core.maximize)

        opt = SolverFactory('mindtpy')
        opt.solve(model)
        # model.display()
        return model.OBJ()

    def evaluate(self, data_path: str, agent: Agent, gamma: float, include_confounder: bool = True,
                 maximize: bool = False) -> float:
        assert gamma >= 1, "Gamma must be >= 1"
        # Read the data
        df = pd.read_csv(data_path)
        columns = df.columns
        state_features = list(filter(lambda c: len(c) < 5 and 'S' in c and "'" not in c, columns))
        human_features = list(filter(lambda c: len(c) < 5 and 'X' in c, columns))
        confounder_features = state_features.copy()
        confounder_features.extend(human_features)
        action_features = list(filter(lambda c: len(c) < 5 and 'A' in c, columns))
        state_action_features = state_features.copy() if not include_confounder else confounder_features.copy()
        state_action_features.extend(action_features)
        # Extra data about next state for ease of use
        next_state_features = list(filter(lambda c: len(c) < 5 and 'S\'' in c, columns))
        terminal_features = next_state_features.copy()
        terminal_features.append("TERMINAL")
        next_state_action_features = state_action_features.copy()
        next_state_action_features.extend(next_state_features)
        # Get the horizon set (set of unique state-action pairs) and the next_horizen_set (S, A, S')
        horizon_set = df[state_action_features]
        horizon_set = horizon_set.groupby(horizon_set.columns.tolist(), as_index=False).size()
        next_horizon_set = df[next_state_action_features]
        next_horizon_set: pd.DataFrame = next_horizon_set.groupby(next_horizon_set.columns.tolist(),
                                                                  as_index=False).size()
        next_horizon_set.insert(len(next_horizon_set.columns), "t_lower", [0 for _ in range(len(next_horizon_set))],
                                True)
        next_horizon_set.insert(len(next_horizon_set.columns), "t_upper", [1 for _ in range(len(next_horizon_set))],
                                True)
        next_horizon_set.insert(len(next_horizon_set.columns), "t_middle", [0.5 for _ in range(len(next_horizon_set))],
                                True)
        next_horizon_set.insert(len(next_horizon_set.columns), "R", [0 for _ in range(len(next_horizon_set))],
                                True)
        # Get the state set and action set so we can extract |S| and |A|
        state_set = horizon_set[state_features]
        next_state_set = df[next_state_features]
        next_state_set.columns = state_features
        state_set = pd.concat([state_set, next_state_set]).drop_duplicates(ignore_index=True)
        action_set = horizon_set[action_features].drop_duplicates(ignore_index=True)
        terminal_set = df[terminal_features].drop_duplicates(ignore_index=True)
        terminal = list(terminal_set["TERMINAL"])
        S = len(state_set)
        A = len(action_set)
        # Get observed behavioral policy from data (action taken from state divided by visits to state)
        behavioral_policy = horizon_set.copy()
        for index, row in behavioral_policy.iterrows():
            state = row[state_features]
            total = horizon_set[np.all(horizon_set[state_features] == state, axis=1)]['size'].sum()
            row['size'] /= total
            behavioral_policy.iloc[index] = row
        # Get possible models
        for index, row in next_horizon_set.iterrows():
            state = row[state_features]
            action = row[action_features]
            r = df[np.all(df[state_features] == state, axis=1) & np.all(df[action_features] == action, axis=1)][
                'R'].mean()
            n_s = horizon_set[np.all(horizon_set[state_features] == state, axis=1)]['size'].sum()
            n_sa = horizon_set[np.all(horizon_set[state_features] == state, axis=1)
                               & np.all(horizon_set[action_features] == action, axis=1)]['size'].sum()
            pi_sa = behavioral_policy[np.all(behavioral_policy[state_features] == state, axis=1)
                                      & np.all(behavioral_policy[action_features] == action, axis=1)]['size'].sum()
            row['t_middle'] = row['size'] / n_sa
            delta_pi = math.sqrt(math.log(2 * S * A / self.delta1) / (2 * n_s))
            delta_tr = math.sqrt(math.log(2 * S * S * A / self.delta2) / (2 * n_sa))
            alpha = 1 / gamma - (1 - 1 / gamma) * (pi_sa + delta_pi)
            beta = gamma + (1 - gamma) * (pi_sa + delta_pi)
            row['t_lower'] = max(0, alpha * (row['t_middle'] - delta_tr))
            row['t_upper'] = min(1, beta * (row['t_middle'] + delta_tr))
            row['R'] = r
            next_horizon_set.iloc[index] = row
        # Find the model that minimizes value function
        mdp = self.find_mdp(df, next_horizon_set,
                            create_feature_dict(state_features, action_features,
                                                confounder_features, state_action_features,
                                                next_state_features),
                            agent=agent,
                            state_set=state_set,
                            action_set=action_set, terminal=terminal, maximize=maximize)
        # Return minimized reward
        return mdp


if __name__ == '__main__':
    evaluator = Evaluator(0.025, 0.025)
    evaluator.evaluate("../4x4_maze_data.csv", None, 3, False)
