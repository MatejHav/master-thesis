import pandas as pd

from models import *
from models import Agent


class Evaluator:

    def __init__(self, behavioral_policy_delta: float, transitional_delta: float):
        assert 0 < behavioral_policy_delta < 1 and 0 < transitional_delta < 1, "Delta must be less than 1 and greater than 0"
        self.delta1 = behavioral_policy_delta
        self.delta2 = transitional_delta

    def evaluate(self, data_path: str, agent: Agent, gamma: float, include_confounder: bool = True) -> float:
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
        next_state_features = state_action_features.copy()
        next_state_features.extend(list(filter(lambda c: len(c) < 5 and 'S\'' in c, columns)))
        # Get the horizon set (set of unique state-action pairs) and the next_horizen_set (S, A, S')
        horizon_set = df[state_action_features]
        horizon_set = horizon_set.groupby(horizon_set.columns.tolist(), as_index=False).size()
        next_horizon_set = df[next_state_features]
        next_horizon_set: pd.DataFrame = next_horizon_set.groupby(next_horizon_set.columns.tolist(), as_index=False).size()
        next_horizon_set.insert(len(next_horizon_set.columns), "t_lower", [0 for _ in range(len(next_horizon_set))], True)
        next_horizon_set.insert(len(next_horizon_set.columns), "t_upper", [1 for _ in range(len(next_horizon_set))], True)
        next_horizon_set.insert(len(next_horizon_set.columns), "t_middle", [0.5 for _ in range(len(next_horizon_set))], True)
        # Get the state set and action set so we can extract |S| and |A|
        state_set = horizon_set[state_features].drop_duplicates(ignore_index=True)
        action_set = horizon_set[action_features].drop_duplicates(ignore_index=True)
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
            n_s = horizon_set[np.all(horizon_set[state_features] == state, axis=1)]['size'].sum()
            n_sa = horizon_set[np.all(horizon_set[state_features] == state, axis=1)
                              & np.all(horizon_set[action_features] == action, axis=1)]['size'].iloc[0]
            pi_sa = behavioral_policy[np.all(behavioral_policy[state_features] == state, axis=1)
                              & np.all(behavioral_policy[action_features] == action, axis=1)]['size'].iloc[0]
            row['t_middle'] = row['size'] / n_sa
            delta_pi = math.sqrt(math.log(2 * S * A / self.delta1) / (2 * n_s))
            delta_tr = math.sqrt(math.log(2 * S * S * A / self.delta2) / (2 * n_sa))
            alpha = 1 / gamma - (1 - 1/gamma) * (pi_sa + delta_pi)
            beta = gamma + (1 - gamma) * (pi_sa + delta_pi)
            row['t_lower'] = max(0, alpha * (row['t_middle'] - delta_tr))
            row['t_upper'] = min(1, beta * (row['t_middle'] + delta_tr))
            next_horizon_set.iloc[index] = row
        # Find the model that minimizes value function
        
        # Return minimized reward
        return 0.0


if __name__ == "__main__":
    evaluator = Evaluator(0.025, 0.025)
    evaluator.evaluate("../4x4_maze_data.csv", None, 3, False)
