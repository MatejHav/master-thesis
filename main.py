import numpy as np
import matplotlib.pyplot as plt

from data.generators import *
from data.builders import *
from environments import *
from models.Experiment import *

if __name__ == '__main__':
    # m, n = 5, 5
    # default_r = -0.01
    # max_r = 10
    # p = 0.2
    mdp_builder, mdp = build_basic_mdp()
    # mdp.visualize(max_iter=100)
    # Human
    # F1 = Function(lambda: np.random.rand() / 4)
    # X1 = Variable(F1)
    X1 = Constant(0.5, is_continuous=False)
    human_features = [X1]
    # Generator
    generator = Generator(mdp, human_features)
    path = "basic_maze_data.csv"
    df = generator.generate_uniform_data(num_of_rows=200, n_jobs=10,
                                         starting_state=lambda _: mdp_builder.get_state("S(0, 0)"), max_iter=5,
                                         verbose=1)
    df.to_csv(path)
    max_epochs = 70
    action_list = mdp_builder.action_list
    print(action_list)
    experiment = Experiment(state_size=3, action_size=1, pos_actions=action_list, x_size=1, gamma=0.5)
    means = []
    losses = []
    for epoch in range(max_epochs):
        print(epoch)
        loss = experiment.train(path, verbose=0)
        losses.append(loss)
        res = experiment.evaluate(mdp, mdp_builder.get_state("S(0, 0)"), action_list, human_features, num_of_episodes=1, max_iter=5)
        means.append(np.mean(res))
    plt.plot(means)
    plt.title("Average return with learned policy")
    plt.ylabel("Epoch number")
    plt.xlabel("Average policy return")
    plt.show()
    plt.plot(losses)
    plt.title("Average loss of training data")
    plt.show()


