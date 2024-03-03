import pyomo.core
from pyomo.environ import *
import numpy as np

import pandas as pd

def create_msm_model(data, gamma, treatement, is_lower_bound):
    # Data according to treated
    all_data = data
    data = data[data["T0"] == treatement]

    # Get distinct x values - potentially group them
    x_features = list(filter(lambda c: "X" in c, data.columns))
    X = data[data["T0"] == treatement].groupby(x_features, as_index=False).size()
    x_size = len(X)

    # Get distinct y values - potentially group them
    Y = data.groupby(["Y0"], as_index=False).size()
    y_size = len(Y)

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
    y_probabilities = all_data.groupby(x_and_y_features, as_index=False).size()
    y_probabilities["probability"] = y_probabilities["size"] / sum(y_probabilities["size"])

    def propensity(x_index):
        x = X.iloc[x_index]
        return propensity_scores[np.all(propensity_scores[x_features] == x[x_features], axis=1)]["propensity_score"].iloc[0]

    def p(y_index, x_index):
        x = X.iloc[x_index]
        y = Y.iloc[y_index]
        selection = y_probabilities[np.all(y_probabilities[x_features] == x[x_features], axis=1) & np.all(y_probabilities[["Y0"]] == y[["Y0"]], axis=1)]
        if len(selection) == 0:
            return 0
        return selection["probability"].iloc[0]

    model = ConcreteModel(name="MarginalSensitivityModel")
    model.X = RangeSet(0, x_size-1)
    model.Y = RangeSet(0, y_size-1)
    model.lam = Var(model.X, model.Y, bounds=(1 / gamma, gamma))

    # Constraint 1: Lambda is a distribution of Y | X, T
    def distribution_constraint(model):
        return sum([sum(model.lam[x, y] for y in model.Y) for x in model.X]) / (x_size * y_size) == 1

    model.c1 = Constraint(rule=distribution_constraint)

    # Constraint 2: Propensity scores remains unchanged
    def propensity_constraint(model, x):
        return sum([propensity(x) * model.lam[x, y] for y in model.Y]) == propensity(x)

    model.c2 = Constraint(model.X, rule=propensity_constraint)

    # Constraint 3: Distribution of Y unchanged
    def p_constraint(model, y):
        return sum([p(y, x) * model.lam[x, y] for x in model.X]) == sum(
            [p(y, x) for x in model.X])

    model.c3 = Constraint(model.Y, rule=p_constraint)

    # Objective: Find the bound
    def objective_function(model):
        return sum([sum([y * model.lam[x, y] for y in model.Y]) for x in model.X]) / (y_size * x_size)

    model.OBJ = Objective(rule=objective_function,
                          sense=pyomo.core.minimize if is_lower_bound else pyomo.core.maximize)

    from pyomo.util.model_size import build_model_size_report

    report = build_model_size_report(model)

    print('Num constraints: ', report.activated.constraints)

    print('Num variables: ', report.activated.variables)
    # exit()


    opt = SolverFactory('ipopt')
    opt.options['max_iter'] = 5000
    # opt.options['acceptable_tol'] = 0.000001
    opt.solve(model)
    # model.display()
    return model.OBJ()


def create_f_sensitivity_model(data, rho, treated, lower_bound, f):
    # Data according to treated
    data = data[data["T"] == treated]

    # Get distinct x values - potentially group them
    x_features = list(filter(lambda c: "X" in c, data.columns))
    X = data[x_features]
    x_size = len(X.unique())

    # Get distinct y values - potentially group them
    Y = data["Y"]
    y_size = len(Y.unique())

    model = ConcreteModel(name="FSensitivityModel")
    model.X = RangeSet(0, x_size)
    model.Y = RangeSet(0, y_size)
    model.L = Var(model.X, model.Y, bounds=(0, 1))
    model.R = Var(model.X, bounds=(-50, 50))

    # Constraint 1: Definition of R
    def r_constraint(model, x):
        return model.R[x] == 0

    model.c1 = Constraint(model.X, rule=r_constraint)

    # Constraint 2: MSM assumption
    def msm_constraint(model, x):
        return f(sum([model.L[x, y] / model.R[x] for y in model.Y])) / y_size <= rho

    model.c2 = Constraint(model.state, model.action, rule=msm_constraint)

    # Objective: V(s_0)
    def objective_function(model):
        return sum([sum([y * model.L(x, y) for y in model.Y]) for x in model.X]) / (y_size * x_size)

    model.OBJ = Objective(rule=objective_function,
                          sense=pyomo.core.minimize if lower_bound else pyomo.core.maximize)

    opt = SolverFactory('ipopt')
    opt.options['max_iter'] = 5000
    # opt.options['acceptable_tol'] = 0.000001
    opt.solve(model)
    # model.display()
    return model.OBJ()

if __name__ == '__main__':
    df = pd.read_csv("../csv_files/regular.csv")
    res = create_msm_model(df, 1, 1, True)
    print(res)