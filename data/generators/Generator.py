import threading
import pandas as pd

from tqdm import tqdm
from data.generators.MDP import *


class Generator:

    def __init__(self, mdp: MDP, x: List[Value]):
        self.mdp = mdp
        self.x = x

    def generate_uniform_data(self, num_of_rows: int, n_jobs: int, starting_state: Callable[List[Value], State], max_iter: int=100):
        rows_per_thread = [num_of_rows // n_jobs for _ in range(n_jobs)]
        remainder = num_of_rows % n_jobs
        for i in range(n_jobs):
            if remainder == 0:
                break
            rows_per_thread[i] += 1
            remainder -= 1
        columns = [f"X{i}" for i in range(len(self.x))]
        columns.append("ID")
        columns.extend([f"S{i}" for i in range(self.mdp.state_size)])
        columns.append("T")
        columns.append("R")
        columns.extend([f"A{i}" for i in range(self.mdp.action_size)])
        result_table = []

        def thread_helper(num_of_rows, thread_num):
            bar = tqdm(range(num_of_rows))
            bar.set_description(f"[THREAD {thread_num}]")
            for i in bar:
                c = 0
                x = [X() for X in self.x]
                state = starting_state(self.x)
                map(lambda X: X.reset(), self.x)
                while state is not None and c <= max_iter:
                    pos_actions = list(self.mdp.get_actions(state).keys())
                    if len(pos_actions) == 0:
                        break
                    A = np.random.choice(pos_actions)
                    row = x.copy()
                    row.append(i)
                    for s in state.X(A):
                        row.append(s())
                        s.reset()
                    row.append(c)
                    row.append(state.R)
                    for a in A.a:
                        row.append(a())
                        a.reset()
                    result_table.append(row)
                    c += 1
                    state = self.mdp.perform_action(state, A, self.x)

        threads = []
        for job in range(n_jobs):
            thread = threading.Thread(target=thread_helper, args=[rows_per_thread[job], job])
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        return pd.DataFrame(result_table, columns=columns)