import pandas as pd

from models import *
from models import Agent


class Evaluator:

    def __init__(self, delta: float):
        assert 0 < delta < 1, "Delta must be less than 1 and greater than 0"
        self.delta = delta

    def evaluate(self, data_path: str, agent: Agent, include_confounder: bool = True) -> float:
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
        # Get the horizon set (set of unique state-action pairs)
        horizon_set = df[state_action_features]
        horizon_set = horizon_set.groupby(horizon_set.columns.tolist(), as_index=False).size()
        next_horizon_set = df[next_state_features]
        next_horizon_set: pd.DataFrame = next_horizon_set.groupby(next_horizon_set.columns.tolist(), as_index=False).size()
        next_horizon_set.insert(len(next_horizon_set.columns), "t_lower", [0 for _ in range(len(next_horizon_set))], True)
        next_horizon_set.insert(len(next_horizon_set.columns), "t_upper", [1 for _ in range(len(next_horizon_set))], True)
        next_horizon_set.insert(len(next_horizon_set.columns), "t_middle", [0.5 for _ in range(len(next_horizon_set))], True)
        # Get possible models
        for index, row in next_horizon_set.iterrows():
            state = row[state_features]
            action = row[action_features]
            total = horizon_set[np.all(horizon_set[state_features] == state, axis=1)
                              & np.all(horizon_set[action_features] == action, axis=1)]['size'].iloc[0]
            row['t_middle'] = row['size'] / total
            row['t_lower'] = max(0, row['t_middle'] - math.sqrt(math.log(2 / self.delta) / (2 * row['size'])))
            row['t_upper'] = min(1, row['t_middle'] + math.sqrt(math.log(2 / self.delta) / (2 * row['size'])))
            next_horizon_set.iloc[index] = row
        print(next_horizon_set)
        # TODO: Find the model that minimizes agents return

        # Return minimized reward
        return 0.0


if __name__ == "__main__":
    evaluator = Evaluator(0.95)
    evaluator.evaluate("../4x4_maze_data.csv", None, False)
