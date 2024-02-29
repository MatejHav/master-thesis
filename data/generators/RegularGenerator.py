import threading
import pandas as pd
from tqdm import tqdm
class RegularGenerator:

    def __init__(self, generators, noise_generators, sizes):
        self.size = sizes
        self.generators = generators
        self.noise_generators = noise_generators

    def generate(self, num_rows, n_jobs, path, verbose=0):
        data = []
        def _generator_helper(k):
            bar = range(k)
            if verbose > 0:
                bar = tqdm(bar)
            for _ in bar:
                U = self.generators["U"](self.noise_generators["U"]())
                X = self.generators["X"](U, self.noise_generators["X"]())
                T = self.generators["T"](U, X, self.noise_generators["T"]())
                Y = self.generators["Y"](U, X, T, self.noise_generators["Y"]())
                data.append([*U, *X, *T, *Y])

        threads = []
        for i in range(n_jobs):
            thread = threading.Thread(target=_generator_helper, args=[num_rows // n_jobs])
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        df = pd.DataFrame(data, columns=[*[f"U{i}" for i in range(self.size["U"])],
                                         *[f"X{i}" for i in range(self.size["X"])],
                                         *[f"T{i}" for i in range(self.size["T"])],
                                         *[f"Y{i}" for i in range(self.size["Y"])]])
        df.to_csv(path)
        return df
