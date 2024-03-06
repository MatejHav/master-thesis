import numpy as np

from data.generators import *

if __name__ == '__main__':
    # General settings
    n_rows = 1500000
    n_jobs = 30
    sizes = {
        "U": 1,
        "X": 2,
        "T": 1,
        "Y": 1
    }
    p = 0.01

    # Generators
    u_gen = lambda noise: [1 if np.random.rand() >= 1 - p else 0]
    x_gen = lambda u, noise: [1 if np.random.rand() >= 0.5 else 0 for _ in range(sizes["X"])]
    t_gen = lambda u, x, noise: [1 if np.random.rand() + 0.2 * u[0] - 0.1 * x[0] >= 0.6 else 0]
    y_gen = lambda u, x, t, noise: [int(x[0] + u[0] + 2 * t[0])]
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
    path = f"./csv_files/regular_{int(100 * p)}.csv"
    df = generator.generate(num_rows=n_rows, n_jobs=n_jobs, path=path)
    print(df.mean(axis=0))

