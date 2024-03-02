import pyomo.core
from pyomo.environ import *

def create_msm_model(data, gamma, treated, lower_bound):
    # Data according to treated
    data = data[data["T"] == treated]

    # Get distinct x values - potentially group them
    x_features = list(filter(lambda c: "X" in c, data.columns))
    X = data[x_features]
    x_size = len(X.unique())

    # Get distinct y values - potentially group them
    Y = data["Y"]
    y_size = len(Y.unique())

    model = ConcreteModel(name="MarginalSensitivityModel")
    model.X = RangeSet(0, x_size)
    model.Y = RangeSet(0, y_size)
    model.L = Var(model.X, model.Y, bounds=(0, 1))
    model.R = Var(model.X, bounds=(-50, 50))

    # Constraint 1: Definition of R
    def r_constraint(model, x):
        return model.R[x] == 0

    model.c1 = Constraint(model.X, rule=r_constraint)

    # Constraint 2: MSM assumption
    def msm_constraint(model, x, y):
        return 1 / gamma <= model.L[x, y] / model.R[x] <= gamma # TODO - MSM Assumption

    model.c2 = Constraint(model.X, model.Y, rule=msm_constraint)

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
    # print(agent_policy)
    # exit()
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

    model = ConcreteModel(name="MarginalSensitivityModel")
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
    # print(agent_policy)
    # exit()
    return model.OBJ()