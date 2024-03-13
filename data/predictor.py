import pandas as pd
import numpy as np
from causalml.inference.tree import *

if __name__ == '__main__':
    df = pd.read_csv("../csv_files/data.csv")
    x_features = list(filter(lambda c: "X" in c, df.columns))
    t_features = "T0"
    y_features = "Y0"
    X = df[x_features].to_numpy()
    T = df[t_features].to_numpy()
    Y = df[y_features].to_numpy()
    estimator = CausalRandomForestRegressor(min_samples_leaf=20, min_samples_split=20)
    estimator.fit(X, T, Y)
    predictions = estimator.predict(X, with_outcomes=True)

    res = pd.DataFrame(columns=[*x_features, t_features, y_features])
    for i in range(len(X)):
        row0 = [*X[i], 0, int(predictions[i, 0])]
        row1 = [*X[i], 1, int(predictions[i, 1])]
        res.loc[len(res)] = row0
        res.loc[len(res)] = row1
    res.to_csv("../csv_files/data_adjusted.csv")

