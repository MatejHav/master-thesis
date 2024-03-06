import math

import pyomo.core
from pyomo.core import *
from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


def create_msm_model(data, gamma, treatement, is_lower_bound, alpha_prop, alpha_prob):
    # Data according to treated
    all_data = data
    data = data[data["T0"] == treatement]

    # Get distinct x values - potentially group them
    x_features = list(filter(lambda c: "X" in c, data.columns))
    X = data[data["T0"] == treatement].groupby(x_features, as_index=False).size()
    x_size = len(X)
    X["prior"] = X["size"] / sum(X["size"])

    # Get distinct y values - potentially group them
    Y = data.groupby(["Y0"], as_index=False).size()
    y_size = len(Y)
    Y["prior"] = Y["size"] / sum(Y["size"])

    # Propensity scores
    all_data_frequency = all_data.groupby(x_features, as_index=False).size()
    treated_data_frequency = all_data[all_data["T0"] == treatement].groupby(x_features, as_index=False).size()
    treated_data_frequency.rename(columns={"size": "treated_size"}, inplace=True)
    propensity_scores = pd.merge(all_data_frequency, treated_data_frequency, on=x_features, how='left')
    propensity_scores.fillna(0, inplace=True)
    propensity_scores["propensity_score"] = propensity_scores["treated_size"] / propensity_scores["size"]

    # Y Probabilities: P(Y| X, T)
    x_and_y_features = x_features.copy()
    x_and_y_features.append("Y0")
    y_probabilities = data.groupby(x_and_y_features, as_index=False).size()
    y_probabilities["probability"] = y_probabilities["size"] / sum(y_probabilities["size"])

    def propensity(x_index):
        x = X.iloc[x_index]
        return \
        propensity_scores[np.all(propensity_scores[x_features] == x[x_features], axis=1)]["propensity_score"].iloc[0]

    def p(y_index, x_index):
        x = X.iloc[x_index]
        y = Y.iloc[y_index]
        selection = y_probabilities[
            np.all(y_probabilities[x_features] == x[x_features], axis=1) & np.all(y_probabilities[["Y0"]] == y[["Y0"]],
                                                                                  axis=1)]
        if len(selection) == 0:
            return 0
        return selection["size"].iloc[0] / x["size"]

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

    def epsilon_propensity(x_index):
        x = X.iloc[x_index]
        return np.sqrt(1 / x["size"] * np.log(2 / alpha_prop))

    def epsilon_probability(x_index, y_index):
        count = size(y_index, x_index)
        if count == 0:
            return 0
        return np.sqrt(1 / count * np.log(2 * y_size / alpha_prob))

    model = ConcreteModel(name="MarginalSensitivityModel")
    model.X = RangeSet(0, x_size - 1)
    model.Y = RangeSet(0, y_size - 1)
    model.lam = Var(model.X, model.Y, bounds=(1 / gamma, gamma), initialize=1)
    model.propensity = Var(model.X, bounds=(0, 1))
    model.prob = Var(model.X, model.Y, bounds=(0, 1))

    # Constraint 1: Lambda is a distribution of Y | X, T
    def distribution_constraint(model):
        return sum([sum(model.lam[x, y] * model.prob[x, y] for y in model.Y) * X.iloc[x]["size"] for x in model.X]) / sum(X["size"]) == 1

    model.c1 = Constraint(rule=distribution_constraint)

    # Constraint 2: Propensity scores remains unchanged
    def propensity_constraint(model):
        return sum([sum([model.propensity[x] * model.lam[x, y] * model.prob[x, y] for y in model.Y]) for x in model.X]) == sum(
            [sum([model.propensity[x] * model.prob[x, y] for y in model.Y]) for x in model.X])

    model.c2 = Constraint(rule=propensity_constraint)

    # Constraint 3: Distribution of Y unchanged
    def p_constraint(model):
        return sum([sum([model.prob[x, y] * model.lam[x, y] * X.iloc[x]["size"] for x in model.X]) / sum(X["size"]) for y in
                    model.Y]) == sum([sum(
            [model.prob[x, y] * X.iloc[x]["size"] for x in model.X]) / sum(X["size"]) for y in model.Y])

    model.c3 = Constraint(rule=p_constraint)

    # Constraint 4: Propensity definition
    def propensity_definition_constraint(model, x):
        expected_propensity = propensity(x)
        bound = epsilon_propensity(x)
        return max(0, expected_propensity - bound), model.propensity[x], min(1, expected_propensity + bound)

    model.c4 = Constraint(model.X, rule=propensity_definition_constraint)

    # Constraint 5: Probability of Y given X definition
    def probability_of_y_constraint(model, x, y):
        expected_prob = p(y, x)
        bound = epsilon_probability(x, y)
        return max(0, expected_prob - bound), model.prob[x, y], min(1, expected_prob + bound)

    model.c5 = Constraint(model.X, model.Y, rule=probability_of_y_constraint)

    # Constraint 6: Probability over y is 1
    def probability_distribution_constraint(model, x):
        return sum([model.prob[x, y] for y in model.Y]) == 1

    model.c6 = Constraint(model.X, rule=probability_distribution_constraint)

    # Objective: Find the bound
    def objective_function(model):
        return sum([sum([get_value(y) * model.lam[x, y] * model.prob[x, y] for y in model.Y]) * X.iloc[x]["size"] for x in model.X]) / sum(X["size"])

    model.OBJ = Objective(rule=objective_function,
                          sense=pyomo.core.minimize if is_lower_bound else pyomo.core.maximize)

    opt = SolverFactory('ipopt')
    opt.solve(model)
    # model.display()
    # exit()
    return model.OBJ()


def create_f_sensitivity_model(data, rho, treatment, is_lower_bound, alpha_prop, alpha_prob):
    # Data according to treated
    all_data = data
    p_treated = len(data[data["T0"] == treatment]) / len(all_data)
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
    treated_data_frequency = all_data[all_data["T0"] == treatment].groupby(x_features, as_index=False).size()
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
        return propensity_scores[np.all(propensity_scores[x_features] == x[x_features], axis=1)]["propensity_score"].iloc[0]

    def size(y_index, x_index):
        x = X.iloc[x_index]
        y = Y.iloc[y_index]
        selection = y_probabilities[
            np.all(y_probabilities[x_features] == x[x_features], axis=1) & np.all(y_probabilities[["Y0"]] == y[["Y0"]],
                                                                                  axis=1)]
        if len(selection) == 0:
            return 0
        return selection["size"].iloc[0]

    def p(y_index, x_index):
        x = X.iloc[x_index]
        y = Y.iloc[y_index]
        selection = y_probabilities[
            np.all(y_probabilities[x_features] == x[x_features], axis=1) & np.all(y_probabilities[["Y0"]] == y[["Y0"]],
                                                                                  axis=1)]
        if len(selection) == 0:
            return 0
        return selection["size"].iloc[0] / x["size"]

    def get_value(y):
        return Y.iloc[y]["Y0"]

    def epsilon_propensity(x_index):
        x = X.iloc[x_index]
        return np.sqrt(1 / x["size"] * np.log(2 / alpha_prop))

    def epsilon_probability(x_index, y_index):
        count = size(y_index, x_index)
        if count == 0:
            return 0
        return np.sqrt(1 / count * np.log(2 * y_size / alpha_prob))


    model = ConcreteModel(name="FSensitivityModel")
    model.X = RangeSet(0, x_size - 1)
    model.Y = RangeSet(0, y_size - 1)
    model.L = Var(model.X, model.Y, initialize=1, within=NonNegativeReals, bounds=(0, 100))
    model.propensity = Var(model.X, within=NonNegativeReals, bounds=(0, 1))
    model.prob = Var(model.X, model.Y, bounds=(0, 1))

    def r(model, x):
        mean = sum([model.propensity[x] * X.iloc[x]["size"] for x in model.X]) / sum(X["size"])
        return (1 - model.propensity[x]) * mean / (model.propensity[x] * (1 - mean))

    # Constraint 1: Definition of R
    def r_constraint(model, x):
        return r(model, x) == sum([model.L[x, y] * model.prob[x, y] for y in model.Y])

    model.c1 = Constraint(model.X, rule=r_constraint)

    # Constraint 2: MSM assumption
    def f_constraint(model, x):
        return sum([f(model.L[x, y] / r(model, x)) * model.prob[x, y] for y in model.Y]) <= rho

    model.c2 = Constraint(model.X, rule=f_constraint)

    # Constraint 3: Propensity definition
    def propensity_definition_constraint(model, x):
        expected_propensity = propensity(x)
        bound = epsilon_propensity(x)
        return max(0, expected_propensity - bound), model.propensity[x], min(1, expected_propensity + bound)

    model.c3 = Constraint(model.X, rule=propensity_definition_constraint)

    # Constraint 4: Probability of Y given X definition
    def probability_of_y_constraint(model, x, y):
        expected_prob = p(y, x)
        bound = epsilon_probability(x, y)
        return max(0, expected_prob - bound), model.prob[x, y], min(1, expected_prob + bound)

    model.c4 = Constraint(model.X, model.Y, rule=probability_of_y_constraint)

    # Constraint 5: Probability over y is 1
    def probability_distribution_constraint(model, x):
        return sum([model.prob[x, y] for y in model.Y]) == 1

    model.c5 = Constraint(model.X, rule=probability_distribution_constraint)


    # Objective: E[Y * L(X, Y)]
    def objective_function(model):
        return sum([sum([get_value(y) * model.L[x, y] * model.prob[x, y] for y in model.Y]) * X.iloc[x]["size"] for x in model.X]) / sum(X["size"])

    model.OBJ = Objective(rule=objective_function,
                          sense=pyomo.core.minimize if is_lower_bound else pyomo.core.maximize)

    opt = SolverFactory('ipopt')
    # opt.options['max_iter'] = 1000
    # opt.options['acceptable_tol'] = 0.000001
    opt.solve(model)
    # model.display()
    # exit()
    return model.OBJ()


def bounds_creator(data, sensitivity_model, sensitivity_measure):
    sensitivity = (lambda t, l: create_msm_model(data, gamma=sensitivity_measure, treatement=t, is_lower_bound=l, alpha_prop=0.025, alpha_prob=0.025)) \
        if sensitivity_model == 'msm' else \
        lambda t, l: create_f_sensitivity_model(data, rho=sensitivity_measure, treatment=t, is_lower_bound=l, alpha_prop=0.025, alpha_prob=0.025)
    lower_control_bound = sensitivity(0, True)
    lower_treated_bound = sensitivity(1, True)
    upper_control_bound = sensitivity(0, False)
    upper_treated_bound = sensitivity(1, False)

    # Get data parameters for ATE
    average_y_control = data[data["T0"] == 0]["Y0"].mean()
    average_y_treated = data[data["T0"] == 1]["Y0"].mean()

    # Compute the bounds
    return average_y_control, average_y_treated, lower_control_bound, lower_treated_bound, upper_control_bound, upper_treated_bound


if __name__ == '__main__':
    p = 0.99
    df = pd.read_csv(f"../csv_files/regular_{int(100*p)}.csv")
    lower_res = []
    upper_res = []
    control = []
    treated = []
    gammas = np.linspace(1, 3, 10)
    for gamma in gammas:
        y_control, y_treated, lower_control, lower_treated, upper_control, upper_treated = bounds_creator(df,
                                                                                                          sensitivity_model='msm',
                                                                                                          sensitivity_measure=gamma)
        lower_res.append(lower_treated - upper_control)
        upper_res.append(upper_treated - lower_control)
        control.append([lower_control, upper_control])
        treated.append([lower_treated, upper_treated])
    control = np.array(control)
    treated = np.array(treated)
    plt.plot(gammas, lower_res, label='lower bound ATE')
    plt.plot(gammas, upper_res, label='upper bound ATE')
    plt.title(f'Bounds for the ATE generated by MSM, P(U=1)={p}')
    plt.xlabel('Gamma')
    plt.ylabel('ATE')
    plt.ylim(0, 4)
    plt.legend()
    plt.savefig(f"../msm_ate_{int(100*p)}_{len(df)}_uncertain.png")
    plt.show()
    plt.plot(gammas, control[:, 0], label='lower bound')
    plt.plot(gammas, control[:, 1], label='upper bound')
    plt.title(f'Average control Y generated by MSM, P(U=1)={p}')
    plt.xlabel('Gamma')
    plt.ylabel('E[Y | T=0]')
    plt.legend()
    plt.savefig(f"../msm_control_{int(100*p)}_{len(df)}_uncertain.png")
    plt.show()
    plt.plot(gammas, treated[:, 0], label='lower bound')
    plt.plot(gammas, treated[:, 1], label='upper bound')
    plt.title(f'Average treated Y generated by MSM, P(U=1)={p}')
    plt.xlabel('Gamma')
    plt.ylabel('E[Y | T=1]')
    plt.legend()
    plt.savefig(f"../msm_treated_{int(100*p)}_{len(df)}_uncertain.png")
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
    plt.ylim(0, 4)
    plt.legend()
    plt.savefig(f"../fm_ate_{int(100*p)}_{len(df)}_uncertain.png")
    plt.show()
    plt.plot(rhos, control[:, 0], label='lower bound')
    plt.plot(rhos, control[:, 1], label='upper bound')
    plt.title(f'Average control Y generated by f-sensitivity model, P(U=1)={p}')
    plt.xlabel('Rho')
    plt.ylabel('E[Y | T=0]')
    plt.legend()
    plt.savefig(f"../fm_control_{int(100*p)}_{len(df)}_uncertain.png")
    plt.show()
    plt.plot(rhos, treated[:, 0], label='lower bound')
    plt.plot(rhos, treated[:, 1], label='upper bound')
    plt.title(f'Average treated Y generated by f-sensitivity model, P(U=1)={p}')
    plt.xlabel('Rho')
    plt.ylabel('E[Y | T=1]')
    plt.legend()
    plt.savefig(f"../fm_treated_{int(100*p)}_{len(df)}_uncertain.png")
    plt.show()
