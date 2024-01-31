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
    mdp_builder, mdp = build_2_phase_treatment_mdp()
    # mdp.visualize(max_iter=100)
    # if os.path.exists(path):
    #     return mdp, mdp_builder

    # Generator
    behavioral_policy = lambda state, pos_actions, x: np.random.choice(pos_actions) if x[0] == 1 else np.random.choice(pos_actions, p=[0.6, 0.4])
    generator = Generator(mdp, human_features, behavioral_policy)
    df = generator.generate_uniform_data(num_of_rows=4000, n_jobs=16,
                                         starting_state=lambda _: mdp_builder.get_state("start"), max_iter=5,
                                         verbose=0)
    df.to_csv(path)
    return mdp, mdp_builder


def one_run(mdp, mdp_builder, agent, path, human_features, max_epochs=20):
    action_list = mdp_builder.action_list
    experiment = Experiment(agent=agent, state_size=mdp.state_size, action_size=mdp.action_size, x_size=1)
    means = []
    losses = []
    for epoch in range(max_epochs):
        loss = experiment.train(path, verbose=0)
        losses.append(loss)
        res = experiment.evaluate(mdp, mdp_builder.get_state("start"), action_list, human_features, num_of_episodes=5,
                                  max_iter=25)
        means.append(np.mean(res))
    return experiment, loss, losses


def main():
    # Human
    F1 = Function(lambda: 1 if np.random.rand() >= 0.25 else 0)
    X1 = Variable(F1)
    # X1 = Constant(1-0.025, is_continuous=False)
    human_features = [X1]

    # RL

    total_runs = 50
    max_epochs = 1
    path = "simple_case_confounding"
    agents = []
    all_losses = []
    for run in tqdm(range(total_runs)):
        temp_path = path + f"{run}.csv"
        mdp, builder = generate_data(path=temp_path, human_features=human_features)
        # agent1 = CustomAgent("Custom Agent1", pick_probability=lambda state: [0.01, 0.01, 0.01, 0.97] if state[0] == 3 else [0.01, 0.01, 0.97, 0.01])
        # agent2 = CustomAgent("Custom Agent2", pick_probability=lambda state: [0.97, 0.01, 0.01, 0.01] if state[1] == 0 else [0.01, 0.01, 0.97, 0.01])
        # agent = QLearningAgent(name="QLearner", state_size=mdp.state_size, actions=action_list, alpha=0.01, gamma=0.95, hidden_dim=10, num_layers=3)
        agent1 = CustomAgent("Custom Agent1", pick_probability=lambda state: [0.9, 0.1] if state[0] == 0 else [0.5, 0.5])
        agent2 = CustomAgent("Custom Agent2", pick_probability=lambda state: [0.1, 0.9] if state[0] == 0 else [0.6, 0.4])
        exp1, _, losses1 = one_run(mdp, builder, agent1, temp_path, human_features=human_features, max_epochs=max_epochs)
        exp2, _, losses2 = one_run(mdp, builder, agent2, temp_path, human_features=human_features, max_epochs=max_epochs)
        agents.append((exp1.agent, exp2.agent))
        all_losses.append((losses1, losses2))

    num_of_gammas = 21
    # Sensitivity analysis
    gammas = np.linspace(1, 15, num_of_gammas)
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
    plt.title(f"Comparison over {total_runs} runs of 2 policies")
    plt.xlabel("Gamma (sensitivity)")
    plt.ylabel("V(s_0)")

    plt.plot(gammas, res[:, :, 0].mean(axis=1), color=BASE_COLORS[0], label=f"Worst case agent {0}")
    plt.fill_between(gammas, res[:, :, 0].mean(axis=1), res[:, :, 0].std(axis=1), alpha=0.2, color=BASE_COLORS[0])
    plt.fill_between(gammas, res[:, :, 0].mean(axis=1), res[:, :, 0].std(axis=1), alpha=0.2, color=BASE_COLORS[0])
    plt.plot(gammas, res[:, :, 1].mean(axis=1), color=BASE_COLORS[1], label=f"Worst case agent {1}")
    plt.fill_between(gammas, res[:, :, 1].mean(axis=1), res[:, :, 1].std(axis=1), alpha=0.2, color=BASE_COLORS[1])
    plt.fill_between(gammas, res[:, :, 1].mean(axis=1), res[:, :, 1].std(axis=1), alpha=0.2, color=BASE_COLORS[1])
    lowest_y = res.min()

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
    plt.fill_between(gammas, res[:, :, 0].mean(axis=1), res[:, :, 0].std(axis=1), alpha=0.2, color=BASE_COLORS[0])
    plt.fill_between(gammas, res[:, :, 0].mean(axis=1), res[:, :, 0].std(axis=1), alpha=0.2, color=BASE_COLORS[0])
    plt.plot(gammas, res[:, :, 1].mean(axis=1), '--', color=BASE_COLORS[1], label=f"Best case agent {1}")
    plt.fill_between(gammas, res[:, :, 1].mean(axis=1), res[:, :, 1].std(axis=1), alpha=0.2, color=BASE_COLORS[1])
    plt.fill_between(gammas, res[:, :, 1].mean(axis=1), res[:, :, 1].std(axis=1), alpha=0.2, color=BASE_COLORS[1])
    highest_y = res.max()

    plt.ylim(lowest_y, highest_y)
    plt.autoscale(False)
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
