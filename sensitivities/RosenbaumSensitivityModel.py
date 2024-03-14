import pandas as pd
from sensitivities import *


class RosenbaumSensitivityModel(SensitivityModel):

    @staticmethod
    def sensitivity_measure(data: pd.DataFrame):
        """
        Returns the sensitivity metric observed based on data given. This data needs to follow the naming convention for
         features (Xi), unobserved features (Ui), treatment (Ti) and outcome (Yi).
        :param data: Dataframe containing the data with correctly named columns.
        :return: The worst case odds ratio computed based on [Rosenbaum 2002a]
        """
        u_features = list(filter(lambda c: "U" in c, data.columns))
        x_features = list(filter(lambda c: "X" in c, data.columns))
        u_and_x_features = u_features.copy()
        u_and_x_features.extend(x_features)
        t_features = list(filter(lambda c: "T" in c, data.columns))
        prop_features = u_and_x_features.copy()
        prop_features.extend(t_features)
        F = data.groupby(u_and_x_features).size()
        T = data.groupby(t_features).size()
        P = data.groupby(prop_features).size()
        # For every X, U pair check the worst odds ratio against a different X, U pair.
        for f in range(len(F)):
            for t in range(len(T)):



    def find_bound(self, sensitivity_measure: float, delta: float, data_uncertaintgity: bool):
        pass