import numpy as np
import matplotlib.pyplot as plt

from data.generators import *

def create_generator(u_prob, x_effect, t_effect, y_effect):
    # General settings
    n_rows = 1500000
    n_jobs = 30
    sizes = {
        "U": 1,
        "X": 1,
        "T": 1,
        "Y": 1
    }
    # u_prob = 0.01
    base_x_prob = 0.5
    # x_effect = 0.1
    base_t_prob = 0.5
    # t_effect = 0.2
    # y_effect = 2

    # Generators
    u_gen = lambda noise: [0 if np.random.rand() >= u_prob else 1]
    x_gen = lambda u, noise: [0 if np.random.rand() >= x_effect * u[0] + base_x_prob else 1 for _ in range(sizes["X"])]
    t_gen = lambda u, x, noise: [0 if np.random.rand() >= t_effect * u[0] - 0.1 * x[0] + base_t_prob else 1]
    y_gen = lambda u, x, t, noise: [x[0] + y_effect * u[0] + 2 * t[0]]
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
    path = f"./csv_files/data_u{int(100*u_prob)}_x{int(100*x_effect)}_t{int(100*t_effect)}_y{int(100*y_effect)}.csv"
    df = generator.generate(num_rows=n_rows, n_jobs=n_jobs, path=path)
    return df

if __name__ == '__main__':
    ps = np.linspace(0, 1, 10)
    xs = np.linspace(-0.5, 0.5, 10)
    ts = np.linspace(-0.5, 0.5, 10)
    ys = np.linspace(-0.5, 0.5, 10)
    # for p in ps:
    #     for x_effect in xs:
    #         for t_effect in ts:
    #             for y_effect in ys:
    #                 create_generator(p, x_effect, t_effect, y_effect)
    c = lambda p, v: 1 - (0.5 * (v + 0.5) + 0.5 * p)

    fig, ax = plt.subplots()
    # p to X
    for p in ps:
        for x in xs:
            circle = plt.Circle((p, x), 0.05, facecolor=(c(p, x), p, p), linewidth=1, edgecolor='black')
            ax.add_patch(circle)
    ax.autoscale_view()
    ax.set_title("Simple settings colored based on P(U=1) and effect of U on X")
    ax.set_xlabel("P(U=1)")
    ax.set_ylabel("E[X|U=1] - E[X|U=0]")
    plt.show()
    # p to T
    fig, ax = plt.subplots()
    for p in ps:
        for t in ts:
            circle = plt.Circle((p, t), 0.05, facecolor=(p, c(p, t), p), linewidth=1, edgecolor='black')
            ax.add_patch(circle)
    ax.autoscale_view()
    ax.set_title("Simple settings colored based on P(U=1) and effect of U on T")
    ax.set_xlabel("P(U=1)")
    ax.set_ylabel("E[T|U=1, X] - E[T|U=0, X]")
    plt.show()
    # p to Y
    fig, ax = plt.subplots()
    for p in ps:
        for y in ys:
            circle = plt.Circle((p, y), 0.05, facecolor=(p, p, c(p, y)), linewidth=1, edgecolor='black')
            ax.add_patch(circle)
    ax.autoscale_view()
    ax.set_title("Simple settings colored based on P(U=1) and effect of U on Y")
    ax.set_xlabel("P(U=1)")
    ax.set_ylabel("E[Y|U=1, X, T] - E[Y|U=0, X, T]")
    plt.show()
    # X to T
    fig, ax = plt.subplots()
    for x in xs:
        for t in ts:
            circle = plt.Circle((x, t), 0.05, facecolor=(0.5 - x, 0.5 - t, 0), linewidth=1, edgecolor='black')
            ax.add_patch(circle)
    ax.autoscale_view()
    ax.set_title("Simple settings colored based on effect of U on X and T")
    ax.set_xlabel("E[X|U=1] - E[X|U=0]")
    ax.set_ylabel("E[T|U=1, X] - E[T|U=0, X]")
    plt.show()
    # T to Y
    fig, ax = plt.subplots()
    for t in ts:
        for y in ys:
            circle = plt.Circle((t, y), 0.05, facecolor=(0, 0.5 - t, 0.5 - y), linewidth=1, edgecolor='black')
            ax.add_patch(circle)
    ax.autoscale_view()
    ax.set_title("Simple settings colored based on effect of U on T and Y")
    ax.set_xlabel("E[T|U=1, X] - E[T|U=0, X]")
    ax.set_ylabel("E[Y|U=1, X, T] - E[Y|U=0, X, T]")
    plt.show()
