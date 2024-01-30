import os.path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

BASE_COLORS = list(mcolors.BASE_COLORS.keys())

from data.generators import *
from data.builders import *
from environments import *
from models import Evaluator, CustomAgent
from models.Experiment import *


def generate_data(path, human_features):
    # m, n = 5, 5
    # default_r = -0.01
    # max_r = 10
    # p = 0.2
    mdp_builder, mdp = build_4x4_blank_mdp()
    # mdp.visualize(max_iter=100)
    if os.path.exists(path):
        return mdp, mdp_builder

    # Generator
    generator = Generator(mdp, human_features)
    df = generator.generate_uniform_data(num_of_rows=2000, n_jobs=10,
                                         starting_state=lambda _: mdp_builder.get_state("S(0, 0)"), max_iter=25,
                                         verbose=0)
    df.to_csv(path)
    return mdp, mdp_builder


def one_run(mdp, mdp_builder, agent, path, human_features, max_epochs=20):
    action_list = mdp_builder.action_list
    # agent = CustomAgent("Custom Agent", pick_probability=lambda state: [0, 0, 0, 1] if state[0] == 3 else [0, 0, 1, 0])
    # agent = CustomAgent("Custom Agent", pick_probability=lambda state: [1, 0, 0, 0] if state[1] == 0 else [0, 0, 1, 0])
    # agent = QLearningAgent(name="QLearner", state_size=3, actions=action_list, alpha=0.01, gamma=0.95, hidden_dim=10, num_layers=3)
    experiment = Experiment(agent=agent, state_size=3, action_size=1, x_size=1)
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
    # F1 = Function(lambda: 0.25 + 0.5 * np.random.rand())
    # X1 = Variable(F1)
    X1 = Constant(1-0.025, is_continuous=False)
    human_features = [X1]

    # RL
    total_runs = 10
    max_epochs = 1
    path = "4x4_maze_data_no_conf_small_transition"
    agents = []
    all_losses = []
    for run in tqdm(range(total_runs)):
        temp_path = path + f"{run}.csv"
        mdp, builder = generate_data(path=temp_path, human_features=human_features)
        agent1 = CustomAgent("Custom Agent1", pick_probability=lambda state: [0.01, 0.01, 0.01, 0.97] if state[0] == 3 else [0.01, 0.01, 0.97, 0.01])
        agent2 = CustomAgent("Custom Agent2", pick_probability=lambda state: [0.97, 0.01, 0.01, 0.01] if state[1] == 0 else [0.01, 0.01, 0.97, 0.01])
        exp1, _, losses1 = one_run(mdp, builder, agent1, temp_path, human_features=human_features, max_epochs=max_epochs)
        exp2, _, losses2 = one_run(mdp, builder, agent2, temp_path, human_features=human_features, max_epochs=max_epochs)
        agents.append((exp1.agent, exp2.agent))
        all_losses.append((losses1, losses2))

    num_of_gammas = 5
    # Sensitivity analysis
    gammas = np.linspace(1, 10, num_of_gammas)
    evaluator = Evaluator(0.025, 0.025)
    # Worst case
    res = []
    for gamma in tqdm(gammas):
        values = []
        for index, (agent1, agent2) in enumerate(agents):
            value1 = evaluator.evaluate(path + f"{index}.csv", agent1, gamma, False)
            value2 = evaluator.evaluate(path + f"{index}.csv", agent2, gamma, False)
            values.append((value1, value2))
        res.append(values)
    res = np.array(res)
    plt.title(f"Comparison of {total_runs} learned policies")
    plt.xlabel("Gamma (sensitivity)")
    plt.ylabel("V(s_0)")

    plt.plot(gammas, res[:, :, 0].mean(axis=1), color=BASE_COLORS[0], label=f"Worst case agent {0}")
    plt.fill_between(gammas, res[:, :, 0].mean(axis=1), res[:, :, 0].max(axis=1), alpha=0.5, color=BASE_COLORS[0])
    plt.fill_between(gammas, res[:, :, 0].mean(axis=1), res[:, :, 0].min(axis=1), alpha=0.5, color=BASE_COLORS[0])
    plt.plot(gammas, res[:, :, 1].mean(axis=1), color=BASE_COLORS[1], label=f"Worst case agent {1}")
    plt.fill_between(gammas, res[:, :, 1].mean(axis=1), res[:, :, 1].max(axis=1), alpha=0.5, color=BASE_COLORS[1])
    plt.fill_between(gammas, res[:, :, 1].mean(axis=1), res[:, :, 1].min(axis=1), alpha=0.5, color=BASE_COLORS[1])

    # Best case
    res = []
    for gamma in tqdm(gammas):
        values = []
        for i, (agent1, agent2) in enumerate(agents):
            value1 = evaluator.evaluate(path + f"{i}.csv", agent1, gamma, False, maximize=True)
            value2 = evaluator.evaluate(path + f"{i}.csv", agent2, gamma, False, maximize=True)
            values.append((value1, value2))
        res.append(values)
    res = np.array(res)
    plt.plot(gammas, res[:, :, 0].mean(axis=1), '--', color=BASE_COLORS[0], label=f"Best case agent {0}")
    plt.fill_between(gammas, res[:, :, 0].mean(axis=1), res[:, :, 0].max(axis=1), alpha=0.5, color=BASE_COLORS[0])
    plt.fill_between(gammas, res[:, :, 0].mean(axis=1), res[:, :, 0].min(axis=1), alpha=0.5, color=BASE_COLORS[0])
    plt.plot(gammas, res[:, :, 1].mean(axis=1), '--', color=BASE_COLORS[1], label=f"Best case agent {1}")
    plt.fill_between(gammas, res[:, :, 1].mean(axis=1), res[:, :, 1].max(axis=1), alpha=0.5, color=BASE_COLORS[1])
    plt.fill_between(gammas, res[:, :, 1].mean(axis=1), res[:, :, 1].min(axis=1), alpha=0.5, color=BASE_COLORS[1])
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
