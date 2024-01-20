import pandas as pd
from models.QLearning import *

class Experiment:

    def __init__(self, state_size: int, action_size: int, pos_actions: List[Action], x_size: int, alpha=0.5, gamma=0.95, include_confounders=False):
        self.state_size = state_size
        self.action_size = action_size
        self.x_size = x_size
        self.agent = QLearningAgent(name="QLearner", state_size=state_size + x_size if include_confounders else state_size, actions=pos_actions, alpha=alpha, gamma=gamma, hidden_dim=10, num_layers=3)
        self.include_conf = include_confounders

    def train(self, csv_file: str, verbose: int=1):
        df = pd.read_csv(csv_file)
        ids = df["ID"].unique()
        np.random.shuffle(ids)
        state_ids = [f"S{i}" for i in range(self.state_size)]
        if self.include_conf:
            state_ids.extend([f"X{i}" for i in range(self.x_size)])
        next_state_ids = [f"S'{i}" for i in range(self.state_size)]
        if self.include_conf:
            next_state_ids.extend([f"X{i}" for i in range(self.x_size)])
        action_ids = [f"A{i}" for i in range(self.action_size)]
        total_loss = 0
        total = 0
        bar = ids
        if verbose >= 1:
            bar = tqdm(ids)
        for i in bar:
            current = df[df["ID"] == i]
            t_max = current["T"].max()
            for t in range(t_max):
                curr_temp = current[current["T"] == t].iloc[0]
                loss = self.agent.learn(state=Tensor(curr_temp[state_ids].astype(float)),
                                 action=Tensor(curr_temp[action_ids].astype(float)),
                                 next_state=Tensor(curr_temp[next_state_ids].astype(float)),
                                 reward=curr_temp["R"],
                                 terminal=curr_temp["TERMINAL"])
                total_loss += loss
                total += 1
        return total_loss / total

    def evaluate_episode(self, mdp: MDP, start_state: State, actions: List[Action], X: List[Value], max_iter:int = 100):
        state = start_state
        c = 0
        cum_r = 0
        s = "EPIDOSE: "
        while state is not None and c < max_iter:
            cum_r += state.R
            state_tensor = [x() for x in state.X]
            if self.include_conf:
                state_tensor.extend([x() for x in X])
            state_tensor = Tensor(state_tensor)
            next_action = self.agent.choose(state_tensor)
            # print(state, next_action, end=', ')
            s += str(next_action) + ", "
            state = mdp.perform_action(state, actions[next_action], X)
            if state is None:
                break
            c += 1
        # print(s)
        [x.reset() for x in X]
        self.res.append(cum_r)

    def evaluate(self, mdp: MDP, start_state: State, actions: List[Action], X: List[Value], num_of_episodes: int=100, max_iter:int=100):
        self.res = []
        threads = []
        for episode in range(num_of_episodes):
            thread = threading.Thread(target=self.evaluate_episode, args=(mdp, start_state, actions, X, max_iter))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        return self.res
