import functools
import math
import threading
import time

import pyomo.core
import torch.optim
from pyomo.core import *
from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt
import functools

import pandas as pd

from sklearn.linear_model import QuantileRegressor
from tqdm import tqdm


def create_msm_model(data, gamma, t, is_lower_bound):
    all_data = data
    # Get distinct x values - potentially group them
    x_features = list(filter(lambda c: "X" in c, all_data.columns))
    X = all_data.groupby(x_features, as_index=False).size()
    x_size = len(X)
    X["prior"] = X["size"] / sum(X["size"])

    # Get distinct y values - potentially group them
    Y = all_data.groupby(["Y0"], as_index=False).size()
    y_size = len(Y)
    Y["prior"] = Y["size"] / sum(Y["size"])

    # Propensity scores
    all_data_frequency = all_data.groupby(x_features, as_index=False).size()
    treated_data_frequency = all_data[all_data["T0"] == 1].groupby(x_features, as_index=False).size()
    treated_data_frequency.rename(columns={"size": "treated_size"}, inplace=True)
    propensity_scores = pd.merge(all_data_frequency, treated_data_frequency, on=x_features, how='left')
    propensity_scores.fillna(0, inplace=True)
    propensity_scores["propensity_score"] = propensity_scores["treated_size"] / propensity_scores["size"]

    # Y Probabilities: P(Y| X, T)
    x_and_y_features = x_features.copy()
    x_and_y_features.append("Y0")
    x_and_y_features.append("T0")
    y_probabilities = all_data.groupby(x_and_y_features, as_index=False).size()
    y_probabilities["probability"] = y_probabilities["size"] / sum(y_probabilities["size"])
    n = sum(y_probabilities["size"])

    size_helper = {}

    def thread_helper(y_index, t):
        for x_index in range(x_size):
            x = X.iloc[x_index]
            y = Y.iloc[y_index]
            size_helper[(t, y_index, x_index)] = y_probabilities[y_probabilities["T0"] == t &
                                                                 np.all(y_probabilities[x_features] == x[x_features],
                                                                        axis=1) & np.all(
                y_probabilities[["Y0"]] == y[["Y0"]],
                axis=1)]

    threads = []
    for i in range(y_size):
        threads.append(threading.Thread(target=thread_helper, args=[i, 0]))
        threads[-1].start()
    for thread in threads:
        thread.join()
    for i in range(y_size):
        threads.append(threading.Thread(target=thread_helper, args=[i, 1]))
        threads[-1].start()
    for thread in threads:
        thread.join()

    def propensity(x_index, t):
        x = X.iloc[x_index]
        return \
            propensity_scores[np.all(propensity_scores[x_features] == x[x_features], axis=1)]["propensity_score"].iloc[
                0]

    def p(y_index, x_index, t):
        x = X.iloc[x_index]
        selection = size_helper[(t, y_index, x_index)]
        if len(selection) == 0:
            return 0
        return selection["size"].iloc[0] / x["size"]

    def size(y_index, x_index, t):
        selection = size_helper[(t, y_index, x_index)]
        if len(selection) == 0:
            return 0
        return selection["size"].iloc[0]

    def get_value(y):
        return Y.iloc[y]["Y0"]

    start_time = time.time()
    model = ConcreteModel(name="MarginalSensitivityModel")
    model.X = RangeSet(0, x_size - 1)
    model.Y = RangeSet(0, y_size - 1)
    model.lam = Var(model.X, model.Y, bounds=(1 / gamma, gamma), initialize=1)

    # Constraint 1: Lambda is a distribution of Y | X, T
    def distribution_constraint(model):
        return sum([sum(model.lam[x, y] * size(y, x, t) for y in model.Y) for x in model.X]) / n == 1

    model.c1 = Constraint(rule=distribution_constraint)

    # Constraint 2: Propensity scores remains unchanged
    def propensity_constraint(model):
        return sum(
            [sum([propensity(x, t) * model.lam[x, y] * size(y, x, t) for y in model.Y]) for x in model.X]) / n == sum(
            [sum([propensity(x, t) * size(y, x, t) for y in model.Y]) for x in model.X]) / n

    model.c2 = Constraint(rule=propensity_constraint)

    # Constraint 3: Distribution of Y unchanged
    def p_constraint(model):
        return sum([sum([p(y, x, t) * model.lam[x, y] * X.iloc[x]["size"] for x in model.X]) / sum(X["size"]) for y in
                    model.Y]) == sum([sum(
            [p(y, x, t) * X.iloc[x]["size"] for x in model.X]) / sum(X["size"]) for y in model.Y])

    model.c3 = Constraint(rule=p_constraint)

    # Objective: Find the bound
    def objective_function(model):
        return sum([sum([get_value(y) * model.lam[x, y] * size(y, x, t) for y in model.Y]) for x in model.X]) / n

    model.OBJ = Objective(rule=objective_function,
                          sense=pyomo.core.minimize if is_lower_bound else pyomo.core.maximize)

    opt = SolverFactory('glpk')
    opt.options['tmlim'] = 100
    opt.solve(model)
    print(f"End: {time.time() - start_time}")
    # model.display()
    return model.OBJ()


def closed_form_msm(data, gamma, is_lower_bound):
    x_features = list(filter(lambda c: 'X' in c, data.columns))
    tau = gamma / (gamma + 1)
    if is_lower_bound:
        tau = 1 - tau
    regressor = QuantileRegressor(quantile=tau, solver='highs')
    regressor.fit(data[x_features], data["Y0"])
    X = data[x_features]
    Y = data["Y0"].to_numpy()
    pred = regressor.predict(X)
    res = 1 / gamma * Y + (1 - 1 / gamma) * (pred + 1 / (1 - tau) * (Y - pred))
    return res.mean()


def closed_form_f_sensitivity(data, rho, treatment, is_lower_bound, eps=1e-8):
    data = data[data["T0"] == treatment]
    outcomes = torch.DoubleTensor(data["Y0"].to_numpy())
    N = len(outcomes)
    alpha = np.exp(-rho)
    z = torch.nn.Parameter(torch.ones(1, dtype=torch.double) * (-1 if is_lower_bound else 1))
    evar = lambda z: (torch.logsumexp(z * outcomes, 0) - np.log(N * alpha)) / z
    optimizer = torch.optim.SGD([z], lr=0.01, maximize=is_lower_bound)
    previous = None
    # Until converged
    while previous is None or abs((previous - z).item()) > eps:
        if (z.item() < 0 and not is_lower_bound) or (z.item() > 0 and is_lower_bound):
            return min(outcomes.max().item(), evar(previous).item()) if not is_lower_bound else\
                max(outcomes.min().item(), evar(previous).item())
        previous = z
        optimizer.zero_grad()
        # Definition from EVaR
        loss = evar(z)
        loss.backward()
        optimizer.step()
    return min(outcomes.max().item(), evar(z).item()) if not is_lower_bound else\
        max(outcomes.min().item(), evar(z).item())


def create_f_sensitivity_model(data, rho, treatment, is_lower_bound):
    all_data = data
    # Get distinct x values - potentially group them
    x_features = list(filter(lambda c: "X" in c, all_data.columns))
    X = all_data.groupby(x_features, as_index=False).size()
    x_size = len(X)
    X["prior"] = X["size"] / sum(X["size"])

    # Get distinct y values - potentially group them
    Y = all_data.groupby(["Y0"], as_index=False).size()
    y_size = len(Y)
    Y["prior"] = Y["size"] / sum(Y["size"])

    # Propensity scores
    all_data_frequency = all_data.groupby(x_features, as_index=False).size()
    treated_data_frequency = all_data[all_data["T0"] == 1].groupby(x_features, as_index=False).size()
    treated_data_frequency.rename(columns={"size": "treated_size"}, inplace=True)
    propensity_scores = pd.merge(all_data_frequency, treated_data_frequency, on=x_features, how='left')
    propensity_scores.fillna(0, inplace=True)
    propensity_scores["propensity_score"] = propensity_scores["treated_size"] / propensity_scores["size"]

    # Y Probabilities: P(Y| X, T)
    x_and_y_features = x_features.copy()
    x_and_y_features.append("Y0")
    x_and_y_features.append("T0")
    y_probabilities = all_data.groupby(x_and_y_features, as_index=False).size()
    y_probabilities["probability"] = y_probabilities["size"] / sum(y_probabilities["size"])
    n = sum(y_probabilities["size"])

    size_helper = {}

    def thread_helper(y_index, t):
        for x_index in range(x_size):
            x = X.iloc[x_index]
            y = Y.iloc[y_index]
            size_helper[(t, y_index, x_index)] = y_probabilities[y_probabilities["T0"] == t &
                                                                 np.all(y_probabilities[x_features] == x[x_features],
                                                                        axis=1) & np.all(
                y_probabilities[["Y0"]] == y[["Y0"]],
                axis=1)]

    threads = []
    for i in range(y_size):
        threads.append(threading.Thread(target=thread_helper, args=[i, 0]))
        threads[-1].start()
    for thread in threads:
        thread.join()
    for i in range(y_size):
        threads.append(threading.Thread(target=thread_helper, args=[i, 1]))
        threads[-1].start()
    for thread in threads:
        thread.join()

    def propensity(x_index, t):
        x = X.iloc[x_index]
        return \
            propensity_scores[np.all(propensity_scores[x_features] == x[x_features], axis=1)]["propensity_score"].iloc[
                0]

    def p(y_index, x_index, t):
        x = X.iloc[x_index]
        selection = size_helper[(t, y_index, x_index)]
        if len(selection) == 0:
            return 0
        return selection["size"].iloc[0] / x["size"]

    def size(y_index, x_index, t):
        selection = size_helper[(t, y_index, x_index)]
        if len(selection) == 0:
            return 0
        return selection["size"].iloc[0]

    def get_value(y):
        return Y.iloc[y]["Y0"]

    # Data according to treated
    all_data = data
    p_treated = len(data[data["T0"] == 1]) / len(all_data)
    data = data[data["T0"] == treatment]
    f = lambda t: t * log(t)

    # Get distinct x values - potentially group them
    x_features = list(filter(lambda c: "X" in c, data.columns))
    X = data[data["T0"] == treatment].groupby(x_features, as_index=False).size()
    x_size = len(X)
    X["prior"] = X["size"] / sum(X["size"])

    # Get distinct y values - potentially group them
    Y = data.groupby(["Y0"], as_index=False).size()
    y_size = len(Y)
    Y["prior"] = Y["size"] / sum(Y["size"])

    # Propensity scores
    all_data_frequency = all_data.groupby(x_features, as_index=False).size()
    treated_data_frequency = all_data[all_data["T0"] == 1].groupby(x_features, as_index=False).size()
    treated_data_frequency.rename(columns={"size": "treated_size"}, inplace=True)
    propensity_scores = pd.merge(all_data_frequency, treated_data_frequency, on=x_features, how='left')
    propensity_scores.fillna(0, inplace=True)
    propensity_scores["propensity_score"] = propensity_scores["treated_size"] / propensity_scores["size"]

    # Y Probabilities: P(Y| X, T)
    x_and_y_features = x_features.copy()
    x_and_y_features.append("Y0")
    y_probabilities = data.groupby(x_and_y_features, as_index=False).size()
    y_probabilities["probability"] = y_probabilities["size"] / sum(y_probabilities["size"])
    n = sum(y_probabilities["size"])

    def propensity(x_index):
        x = X.iloc[x_index]
        return \
        propensity_scores[np.all(propensity_scores[x_features] == x[x_features], axis=1)]["propensity_score"].iloc[0]

    def size(y_index, x_index):
        x = X.iloc[x_index]
        y = Y.iloc[y_index]
        selection = y_probabilities[
            np.all(y_probabilities[x_features] == x[x_features], axis=1) & np.all(y_probabilities[["Y0"]] == y[["Y0"]],
                                                                                  axis=1)]
        if len(selection) == 0:
            return 0
        return selection["size"].iloc[0]

    def get_value(y):
        return Y.iloc[y]["Y0"]

    def r(x):
        return (1 - propensity(x)) * p_treated / (propensity(x) * (1 - p_treated))

    model = ConcreteModel(name="FSensitivityModel")
    model.X = RangeSet(0, x_size - 1)
    model.Y = RangeSet(0, y_size - 1)
    model.L = Var(model.X, model.Y, initialize=1, within=NonNegativeReals)

    # Constraint 1: Definition of R
    def r_constraint(model, x):
        return r(x) == sum([model.L[x, y] * size(y, x) for y in model.Y]) / X.iloc[x]["size"]

    model.c1 = Constraint(model.X, rule=r_constraint)

    # # Constraint 2: R is a distribution
    # def r_distribution_constraint(model):
    #     return sum([model.R[x] for x in model.X]) == 1
    #
    # model.c2 = Constraint(rule=r_distribution_constraint)

    # Constraint 3: MSM assumption
    def f_constraint(model, x):
        return sum([f(model.L[x, y] / r(x)) * size(y, x) for y in model.Y]) / X.iloc[x]["size"] <= rho

    model.c3 = Constraint(model.X, rule=f_constraint)

    # # Constraint 4: L distribution constraint
    # def l_distribution_constraint(model):
    #     return sum([sum([model.L[x, y] for y in model.Y]) for x in model.X]) / (x_size * y_size) == 1
    #
    # model.c4 = Constraint(rule=l_distribution_constraint)

    # Objective: E[Y * L(X, Y)]
    def objective_function(model):
        return sum([sum([get_value(y) * model.L[x, y] * size(y, x) for y in model.Y]) for x in model.X]) / n

    model.OBJ = Objective(rule=objective_function,
                          sense=pyomo.core.minimize if is_lower_bound else pyomo.core.maximize)

    opt = SolverFactory('ipopt')
    opt.options['max_iter'] = 1000
    # opt.options['acceptable_tol'] = 0.000001
    opt.solve(model)
    # model.display()
    # exit()
    return model.OBJ()


def bounds_creator(data, sensitivity_model, sensitivity_measure):
    sensitivity = (lambda t, l: create_msm_model(gamma=sensitivity_measure, t=t, is_lower_bound=l)) \
        if sensitivity_model == 'msm' else \
        lambda t, l: create_f_sensitivity_model(data, rho=sensitivity_measure, treatment=t, is_lower_bound=l)
    lower_control_bound = sensitivity(0, True)
    lower_treated_bound = sensitivity(1, True)
    upper_control_bound = sensitivity(0, False)
    upper_treated_bound = sensitivity(1, False)

    # Get data parameters for ATE
    average_y_control = data[data["T0"] == 0]["Y0"].mean()
    average_y_treated = data[data["T0"] == 1]["Y0"].mean()

    # Compute the bounds
    return average_y_control, average_y_treated, lower_control_bound, lower_treated_bound, upper_control_bound, upper_treated_bound


def main():
    p = 0
    df = pd.read_csv("../csv_files/data_u5_x16_t16_y50.csv")

    # lower = closed_form_msm(df, gamma=1.5, is_lower_bound=True)
    # upper = closed_form_msm(df, gamma=1.5, is_lower_bound=False)
    # print(lower, upper, upper - lower)
    # lower_treated = closed_form_f_sensitivity(df, 0.4, 1, True)
    # upper_treated = closed_form_f_sensitivity(df, 0.4, 1, False)
    # lower_control = closed_form_f_sensitivity(df, 0.4, 0, True)
    # upper_control = closed_form_f_sensitivity(df, 0.4, 0, False)
    # print(lower_control, upper_control, lower_treated, upper_treated)

    lower_res = []
    upper_res = []
    control = []
    treated = []
    gammas = np.linspace(1, 9, 10)
    for gamma in gammas:
        print(gamma)
        y_control, y_treated, lower_control, lower_treated, upper_control, upper_treated = bounds_creator(df,
                                                                                                          sensitivity_model='msm',
                                                                                                          sensitivity_measure=gamma)
        lower_res.append(0.1 * (lower_treated - upper_control))
        upper_res.append(0.1 * (upper_treated - lower_control))
        control.append([0.1 * lower_control, 0.1 * upper_control])
        treated.append([0.1 * lower_treated, 0.1 * upper_treated])
        print(lower_res[-1], upper_res[-1], control[-1], treated[-1])
    control = np.array(control)
    treated = np.array(treated)
    plt.plot(gammas, lower_res, label='lower bound ATE')
    plt.plot(gammas, upper_res, label='upper bound ATE')
    plt.title(f'Bounds for the ATE generated by MSM, P(U=1)={p}')
    plt.xlabel('Gamma')
    plt.ylabel('ATE')
    # plt.ylim(1, 3)
    plt.legend()
    plt.savefig(f"../msm_ate_{int(100 * p)}_{len(df)}.png")
    plt.show()
    plt.plot(gammas, control[:, 0], label='lower bound')
    plt.plot(gammas, control[:, 1], label='upper bound')
    plt.title(f'Average control Y generated by MSM, P(U=1)={p}')
    plt.xlabel('Gamma')
    plt.ylabel('E[Y | T=0]')
    plt.legend()
    plt.savefig(f"../msm_control_{int(100 * p)}_{len(df)}.png")
    plt.show()
    plt.plot(gammas, treated[:, 0], label='lower bound')
    plt.plot(gammas, treated[:, 1], label='upper bound')
    plt.title(f'Average treated Y generated by MSM, P(U=1)={p}')
    plt.xlabel('Gamma')
    plt.ylabel('E[Y | T=1]')
    plt.legend()
    plt.savefig(f"../msm_treated_{int(100 * p)}_{len(df)}.png")
    plt.show()

    # F-sensitivity
    lower_res = []
    upper_res = []
    control = []
    treated = []
    rhos = np.linspace(0, 0.3, 10)
    p_treated = len(df[df["T0"] == 1]) / len(df)
    for rho in rhos:
        y_control, y_treated, lower_control, lower_treated, upper_control, upper_treated = bounds_creator(df,
                                                                                                          sensitivity_model='f',
                                                                                                          sensitivity_measure=rho)
        lower_res.append(lower_treated - upper_control)
        upper_res.append(upper_treated - lower_control)
        control.append([lower_control, upper_control])
        treated.append([lower_treated, upper_treated])
    control = np.array(control)
    treated = np.array(treated)
    plt.plot(rhos, lower_res, label='lower bound ATE')
    plt.plot(rhos, upper_res, label='upper bound ATE')
    plt.title(f'Bounds for the ATE generated by f-sensitivity model, P(U=1)={p}')
    plt.xlabel('Rho')
    plt.ylabel('ATE')
    plt.ylim(1, 3)
    plt.legend()
    plt.savefig(f"../fm_ate_{int(100 * p)}_{len(df)}.png")
    plt.show()
    plt.plot(rhos, control[:, 0], label='lower bound')
    plt.plot(rhos, control[:, 1], label='upper bound')
    plt.title(f'Average control Y generated by f-sensitivity model, P(U=1)={p}')
    plt.xlabel('Rho')
    plt.ylabel('E[Y | T=0]')
    plt.legend()
    plt.savefig(f"../fm_control_{int(100 * p)}_{len(df)}.png")
    plt.show()
    plt.plot(rhos, treated[:, 0], label='lower bound')
    plt.plot(rhos, treated[:, 1], label='upper bound')
    plt.title(f'Average treated Y generated by f-sensitivity model, P(U=1)={p}')
    plt.xlabel('Rho')
    plt.ylabel('E[Y | T=1]')
    plt.legend()
    plt.savefig(f"../fm_treated_{int(100 * p)}_{len(df)}.png")
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv("../csv_files/data_adjusted.csv")
    rhos = np.linspace(0, 2, 20)
    f_upper = []
    f_lower = []
    closed_f_upper = []
    closed_f_lower = []
    for rho in tqdm(rhos):
        # y_control, y_treated, lower_control, lower_treated, upper_control, upper_treated = bounds_creator(df,
        #                                                                                                   sensitivity_model='f',
        #                                                                                                   sensitivity_measure=rho)
        # f_upper.append(upper_treated - lower_control)
        # f_lower.append(lower_treated - upper_control)
        # Closed f sensitivity
        lower_treated = 0.1 * closed_form_f_sensitivity(df, rho, 1, True)
        upper_treated = 0.1 * closed_form_f_sensitivity(df, rho, 1, False)
        lower_control = 0.1 * closed_form_f_sensitivity(df, rho, 0, True)
        upper_control = 0.1 * closed_form_f_sensitivity(df, rho, 0, False)
        print(lower_control, upper_control, lower_treated, upper_treated)
        closed_f_upper.append(upper_treated - lower_control)
        closed_f_lower.append(lower_treated - upper_control)
    # plt.plot(rhos, f_upper, color='red', label='Constraint')
    # plt.plot(rhos, f_lower, color='red')
    plt.plot(rhos, closed_f_upper, color='blue', label='Closed')
    plt.plot(rhos, closed_f_lower, color='blue')
    plt.legend()
    plt.xlabel('Rho')
    plt.ylabel('ATE')
    plt.title('Comparison between closed form and constraints form of f-sensitivity')
    plt.show()
