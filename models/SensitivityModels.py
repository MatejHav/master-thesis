import math

import pyomo.core
from pyomo.core import *
from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


def create_msm_model(data, gamma, treatement, is_lower_bound):
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
    n = sum(y_probabilities["size"])

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

    model = ConcreteModel(name="MarginalSensitivityModel")
    model.X = RangeSet(0, x_size - 1)
    model.Y = RangeSet(0, y_size - 1)
    model.lam = Var(model.X, model.Y, bounds=(1 / gamma, gamma), initialize=1)

    # Constraint 1: Lambda is a distribution of Y | X, T
    def distribution_constraint(model):
        return sum([sum(model.lam[x, y] * size(y ,x) for y in model.Y) for x in model.X]) / n == 1

    model.c1 = Constraint(rule=distribution_constraint)

    # Constraint 2: Propensity scores remains unchanged
    def propensity_constraint(model):
        return sum([sum([propensity(x) * model.lam[x, y] * size(y, x) for y in model.Y]) for x in model.X]) / n == sum(
            [sum([propensity(x) * size(y, x) for y in model.Y]) for x in model.X]) / n

    model.c2 = Constraint(rule=propensity_constraint)

    # Constraint 3: Distribution of Y unchanged
    def p_constraint(model):
        return sum([sum([p(y, x) * model.lam[x, y] * X.iloc[x]["size"] for x in model.X]) / sum(X["size"]) for y in model.Y]) == sum([sum(
            [p(y, x) * X.iloc[x]["size"] for x in model.X]) / sum(X["size"]) for y in model.Y])

    model.c3 = Constraint(rule=p_constraint)

    # Objective: Find the bound
    def objective_function(model):
        return sum([sum([get_value(y) * model.lam[x, y] * size(y, x) for y in model.Y]) for x in model.X]) / n

    model.OBJ = Objective(rule=objective_function,
                          sense=pyomo.core.minimize if is_lower_bound else pyomo.core.maximize)

    opt = SolverFactory('glpk')
    opt.solve(model)
    # model.display()
    return model.OBJ()


def create_f_sensitivity_model(data, rho, treatment, is_lower_bound):
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
    sensitivity = (lambda t, l: create_msm_model(data, gamma=sensitivity_measure, treatement=t, is_lower_bound=l)) \
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


if __name__ == '__main__':
    p = "01"
    df = pd.read_csv(f"../csv_files/regular_{p}.csv")
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
    plt.title(f'Bounds for the ATE generated by MSM, P(U=1)=0.{p}')
    plt.xlabel('Gamma')
    plt.ylabel('ATE')
    plt.legend()
    plt.show()
    plt.plot(gammas, control[:, 0], label='lower bound')
    plt.plot(gammas, control[:, 1], label='upper bound')
    plt.title(f'Average control Y generated by MSM, P(U=1)=0.{p}')
    plt.xlabel('Gamma')
    plt.ylabel('E[Y | T=0]')
    plt.legend()
    plt.show()
    plt.plot(gammas, treated[:, 0], label='lower bound')
    plt.plot(gammas, treated[:, 1], label='upper bound')
    plt.title(f'Average treated Y generated by MSM, P(U=1)=0.{p}')
    plt.xlabel('Gamma')
    plt.ylabel('E[Y | T=1]')
    plt.legend()
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
    plt.title(f'Bounds for the ATE generated by f-sensitivity model, P(U=1)=0.{p}')
    plt.xlabel('Rho')
    plt.ylabel('ATE')
    plt.legend()
    plt.show()
    plt.plot(rhos, control[:, 0], label='lower bound')
    plt.plot(rhos, control[:, 1], label='upper bound')
    plt.title(f'Average control Y generated by f-sensitivity model, P(U=1)=0.{p}')
    plt.xlabel('Rho')
    plt.ylabel('E[Y | T=0]')
    plt.legend()
    plt.show()
    plt.plot(rhos, treated[:, 0], label='lower bound')
    plt.plot(rhos, treated[:, 1], label='upper bound')
    plt.title(f'Average treated Y generated by f-sensitivity model, P(U=1)=0.{p}')
    plt.xlabel('Rho')
    plt.ylabel('E[Y | T=1]')
    plt.legend()
    plt.show()
