import pandas as pd
import numpy as np

from typing import *
from sensitivities import *

class FSensitivityModel(SensitivityModel):
    """
    The f-sensitivity model defined by [Jin 2022] formulates the assumption that measures how close is the odds ratio
    to being constant. Meaning we measure the change in the factor of e(X) / (1 - e(X)) when given information
    about the unobserved confounder. This assumption is bound by ρ.
    """

    def __init__(self, name, f: Callable[[float], float]):
        """
        :param name: Name of the model.
        :param f: f-divergent metric. Function has to be convex.
        """
        super().__init__(name)
        self.f = f
    @staticmethod
    def sensitivity_measure(data: pd.DataFrame):
        """
        Computes the sensitivity measure of a sensitivity model based on given data.
        This metric will be an approximation based on available data.
        The data needs to follow a specific naming convention. Each column containing an unobserved confounder needs
        to have the name "Ui" where i indicates the enumeration of the confounder. Similarly with confounders follow
        convention "Xi", treatments "Ti" (yes the treatment might be a vector) and outcome "Yi". Each of these columns
        needs to be discrete / categorical, as the models will work with only given entries.

        By default, this function uses the KL-divergence function, f(t) = t*log(t)

        :param data: Data used to compute the sensitivity metric.
        :return: Metric found based on data.
        """
        f = lambda t: t * np.log(t)
        u_features = list(filter(lambda c: "U" in c, data.columns))
        x_features = list(filter(lambda c: "X" in c, data.columns))
        u_and_x_features = u_features.copy()
        u_and_x_features.extend(x_features)
        t_features = list(filter(lambda c: "T" in c, data.columns))
        prop_features = u_and_x_features.copy()
        prop_features.extend(t_features)
        X = data.groupby(x_features, as_index=False).size()
        F = data.groupby(u_and_x_features, as_index=False).size()
        T = data.groupby(t_features, as_index=False).size()
        P = data.groupby(prop_features, as_index=False).size()
        # Compute f-sensitivity
        # sum_U(sum_T(f((e(X)/(1-e(X))/(e(X,U)/(1-e(X,U))))) * e(X,U) * P(U) / e(X))) for every X
        worst_metric = 0
        for i in range(len(X)):
            x = X.iloc[i]
            selection = P[np.all(P[x_features] == x[x_features], axis=1)]
            rho = 0
            rho_inv = 0
            for _, row in selection.iterrows():
                treatment = row[t_features]
                observed_propensity = sum(selection[np.all(selection[t_features] == treatment, axis=1)]["size"]) / sum(selection["size"])
                propensity = row["size"] / sum(selection["size"])
                rho += f((propensity / (1 - propensity)) / (observed_propensity / (1 - observed_propensity))) * row["size"] / sum(selection["size"]) * propensity / observed_propensity
                rho_inv += f((observed_propensity / (1 - observed_propensity)) / (propensity / (1 - propensity))) * row["size"] / sum(selection["size"]) * propensity / observed_propensity
            worst_metric = max(worst_metric, rho, rho_inv)
        return worst_metric


    def find_bound(self, sensitivity_measure: float, delta: float, data_uncertainty: bool):
        pass