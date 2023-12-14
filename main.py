import numpy as np
import matplotlib.pyplot as plt

from data.generators import *
from data.builders import *
from environments import *
from models.Experiment import *

if __name__ == '__main__':
    m, n = 5, 5
    default_r = -0.01
    max_r = 10
    p = 0.2
    mdp_builder, mdp = build_maze(m, n, default_r, max_r, p)
    mdp.visualize(max_iter=100)
    # Human
    F1 = Function(lambda: np.random.rand() / 4)
    X1 = Variable(F1)
    human_features = [X1]
    # Generator
    # generator = Generator(mdp, human_features)
    # path = "maze_data.csv"
    # df = generator.generate_uniform_data(num_of_rows=100000, n_jobs=100,
    #                                      starting_state=lambda _: mdp_builder.get_state("S(0, 0)"), max_iter=30,
    #                                      verbose=1)
    # df.to_csv(path)
    # max_epochs = 100
    # experiment = Experiment(state_size=3, action_size=1, pos_actions=mdp_builder.action_list, x_size=1)
    # means = []
    # for epoch in range(max_epochs):
    #     print(epoch)
    #     loss = experiment.train(path)
    #     print(f"Last loss: {loss}")
    #     res = experiment.evaluate(mdp, mdp_builder.get_state("S(0, 0)"), mdp_builder.action_list, human_features, num_of_episodes=10, max_iter=50)
    #     means.append(np.mean(res))
    # plt.plot(means)
    # plt.show()


