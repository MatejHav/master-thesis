import pandas as pd
import numpy as np

from sensitivities import *


class MarginalSensitivityModel(SensitivityModel):
    """
    The Marginal Sensitivity Model (MSM) formulated by [Tan 2006] is formulated around an assumption that the ratio
    between e(X) / (1 - e(X)) will at most change by the factor of Γ when given the information
    about the unobserved confounder.
    """

    @staticmethod
    def sensitivity_measure(data: pd.DataFrame):
        u_features = list(filter(lambda c: "U" in c, data.columns))
        x_features = list(filter(lambda c: "X" in c, data.columns))
        u_and_x_features = u_features.copy()
        u_and_x_features.extend(x_features)
        t_features = list(filter(lambda c: "T" in c, data.columns))
        prop_features = u_and_x_features.copy()
        prop_features.extend(t_features)
        F = data.groupby(u_and_x_features, as_index=False).size()
        T = data.groupby(t_features, as_index=False).size()
        P = data.groupby(prop_features, as_index=False).size()
        # For every X, U pair check the worst odds ratio against a different X, U pair.
        worst_metric = 1
        for f in range(len(F)):
            for t in range(len(T)):
                features = F.iloc[f]
                treatment = T.iloc[t]
                selectionA = P[np.all(P[u_and_x_features] == features[u_and_x_features], axis=1) & np.all(
                    P[t_features] == treatment[t_features], axis=1)]
                if len(selectionA) == 0:
                    continue
                propensity = selectionA["size"].iloc[0] / features["size"]
                # Assume positivity
                if propensity == 1:
                    continue
                # Find observed propensity
                X = P[np.all(P[x_features] == features[x_features], axis=1)]
                selectionB = P[np.all(P[x_features] == features[x_features], axis=1) & np.all(
                    P[t_features] == treatment[t_features], axis=1)]
                if len(selectionB) == 0:
                    continue
                observed_propensity = sum(selectionB["size"]) / sum(X["size"])
                worst_metric = max(worst_metric,
                                   (propensity / (1 - propensity)) / (observed_propensity / (1 - observed_propensity)),
                                   (observed_propensity / (1 - observed_propensity)) / (propensity / (1 - propensity)))
        return worst_metric

    def find_bound(self, sensitivity_measure: float, delta: float, data_uncertainty: bool):
        pass
