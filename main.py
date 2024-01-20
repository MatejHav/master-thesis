import numpy as np
import matplotlib.pyplot as plt

from data.generators import *
from data.builders import *
from environments import *
from models import Evaluator
from models.Experiment import *

if __name__ == '__main__':
    # m, n = 5, 5
    # default_r = -0.01
    # max_r = 10
    # p = 0.2
    mdp_builder, mdp = build_4x4_blank_mdp()
    # mdp.visualize(max_iter=100)
    # Human
    # F1 = Function(lambda: np.random.rand())
    # X1 = Variable(F1)
    X1 = Constant(0.5, is_continuous=False)
    human_features = [X1]
    # Generator
    generator = Generator(mdp, human_features)
    path = "4x4_maze_data.csv"
    # df = generator.generate_uniform_data(num_of_rows=2000, n_jobs=10,
    #                                      starting_state=lambda _: mdp_builder.get_state("S(0, 0)"), max_iter=25,
    #                                      verbose=1)
    # df.to_csv(path)
    max_epochs = 3
    action_list = mdp_builder.action_list
    experiment = Experiment(state_size=3, action_size=1, pos_actions=action_list, x_size=1, gamma=0.5)
    means = []
    losses = []
    for epoch in range(max_epochs):
        loss = experiment.train(path, verbose=0)
        print(epoch, loss)
        losses.append(loss)
        res = experiment.evaluate(mdp, mdp_builder.get_state("S(0, 0)"), action_list, human_features, num_of_episodes=5, max_iter=25)
        means.append(np.mean(res))
    plt.plot(means)
    plt.title("Average return with learned policy")
    plt.ylabel("Epoch number")
    plt.xlabel("Average policy return")
    plt.show()
    plt.plot(losses)
    plt.title("Average loss of training data")
    plt.show()

    evaluator = Evaluator(0.025, 0.025)
    res = evaluator.evaluate("4x4_maze_data.csv", experiment.agent, 3, False)
    print(res)


