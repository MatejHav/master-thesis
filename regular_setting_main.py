import numpy as np

from data.generators import *

if __name__ == '__main__':
    # General settings
    n_rows = 15000
    n_jobs = 15
    sizes = {
        "U": 1,
        "X": 4,
        "T": 1,
        "Y": 1
    }

    # Generators
    u_gen = lambda noise: [np.random.randn() + 0.25 + noise]
    x_gen = lambda u, noise: [np.random.rand() for _ in range(sizes["X"])]
    t_gen = lambda u, x, noise: [1 if u[0] + sum(x) / sizes["X"] >= 0.8 + noise else 0]
    y_gen = lambda u, x, t, noise: [1 + sum(x) - 4 * u[0] + 4 * t[0] + noise]
    generators = {
        "U": u_gen,
        "X": x_gen,
        "T": t_gen,
        "Y": y_gen
    }
    # Noise generators
    noise = {
        "U": lambda: 0,
        "X": lambda: 0,
        "T": lambda: 0,
        "Y": lambda: 0
    }

    generator = RegularGenerator(generators=generators, noise_generators=noise, sizes=sizes)
    path = "./csv_files/regular.csv"
    df = generator.generate(num_rows=n_rows, n_jobs=n_jobs, path=path)
    print(df.mean(axis=0))

