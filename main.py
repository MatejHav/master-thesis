import numpy as np
import matplotlib.pyplot as plt

from data.generators import *
from data.builders import *
from environments import *
from models.Experiment import *

if __name__ == '__main__':
    m, n = 20, 20
    default_r = -0.01
    max_r = 10
    p = 0.2
    mdp_builder, mdp = build_maze(m, n, default_r, max_r, p)
    # Human
    F1 = Function(lambda: np.random.rand() / 4)
    X1 = Variable(F1)
    human_features = [X1]
    # Generator
    generator = Generator(mdp, human_features)
    # df = generator.generate_uniform_data(num_of_rows=10000, n_jobs=50,
    #                                      starting_state=lambda _: mdp_builder.get_state("S(0, 0)"), max_iter=200,
    #                                      verbose=1)
    path = "maze_data.csv"
    # df.to_csv(path)
    max_epochs = 10
    experiment = Experiment(3, 1, 4, 1)
    means = []
    for epoch in range(max_epochs):
        experiment.train(path)
        res = experiment.evaluate(mdp, mdp_builder.get_state("S(0, 0)"), mdp_builder.action_list, human_features, 100, max_iter=200)
        means.append(np.mean(res))
    plt.plot(means)
    plt.show()


