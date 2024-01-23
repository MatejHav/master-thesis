import numpy as np
import matplotlib.pyplot as plt

from data.generators import *
from data.builders import *
from environments import *
from models import Evaluator
from models.Experiment import *

def generate_data(path, human_features):
    # m, n = 5, 5
    # default_r = -0.01
    # max_r = 10
    # p = 0.2
    mdp_builder, mdp = build_4x4_blank_mdp()
    # mdp.visualize(max_iter=100)

    # Generator
    generator = Generator(mdp, human_features)
    df = generator.generate_uniform_data(num_of_rows=2000, n_jobs=10,
                                         starting_state=lambda _: mdp_builder.get_state("S(0, 0)"), max_iter=25,
                                         verbose=0)
    df.to_csv(path)
    return mdp, mdp_builder

def one_run(mdp, mdp_builder, path, human_features, max_epochs=20):
    action_list = mdp_builder.action_list
    experiment = Experiment(state_size=3, action_size=1, pos_actions=action_list, x_size=1, gamma=0.5)
    means = []
    losses = []
    for epoch in range(max_epochs):
        loss = experiment.train(path, verbose=0)
        losses.append(loss)
        res = experiment.evaluate(mdp, mdp_builder.get_state("S(0, 0)"), action_list, human_features, num_of_episodes=5,
                                  max_iter=25)
        means.append(np.mean(res))
    return experiment, loss, losses

def main():
    # Human
    # F1 = Function(lambda: np.random.rand())
    # X1 = Variable(F1)
    X1 = Constant(0.5, is_continuous=False)
    human_features = [X1]

    # RL
    total_runs = 2
    max_epochs = 2
    path = "4x4_maze_data"
    agents = []
    all_losses = []
    for run in tqdm(range(total_runs)):
        temp_path = path + f"{run}.csv"
        mdp, builder = generate_data(path=temp_path, human_features=human_features)
        exp, _, losses = one_run(mdp, builder, temp_path, human_features=human_features, max_epochs=max_epochs)
        agents.append(exp.agent)
        all_losses.append(losses)

    # Sensitivity analysis
    gammas = np.linspace(1, 7, 5)
    evaluator = Evaluator(0.025, 0.025)
    # Worst case
    res = []
    for gamma in gammas:
        values = []
        for index, agent in enumerate(agents):
            value = evaluator.evaluate(f"4x4_maze_data{index}.csv", agent, gamma, False)
            values.append(value)
        res.append(values)
    res = np.array(res)
    plt.title(f"Comparison of {total_runs} learned policies")
    plt.xlabel("Gamma (sensitivity)")
    plt.ylabel("V(s_0)")
    plt.plot(gammas, res.mean(axis=1), color=(0,0,1), labe="Worst case")
    plt.fill_between(res.mean(axis=1), res.max(axis=1), alpha=0.5, color=(0,0,1))
    plt.fill_between(res.mean(axis=1), res.min(axis=1), alpha=0.5, color=(0, 0, 1))

    # Best case
    res = []
    for gamma in gammas:
        values = []
        for agent in agents:
            value = evaluator.evaluate("4x4_maze_data.csv", agent, gamma, False)
            values.append(value)
        res.append(values)
    res = np.array(res)
    plt.plot(gammas, res.mean(axis=1), color=(1,0,0), label="Best case")
    plt.fill_between(res.mean(axis=1), res.max(axis=1), alpha=0.5, color=(1, 0, 0))
    plt.fill_between(res.mean(axis=1), res.min(axis=1), alpha=0.5, color=(1, 0, 0))
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
    # plt.plot(means)
    # plt.title("Average return with learned policy")
    # plt.ylabel("Epoch number")
    # plt.xlabel("Average policy return")
    # plt.show()
    # plt.plot(losses)
    # plt.title("Average loss of training data")
    # plt.show()



