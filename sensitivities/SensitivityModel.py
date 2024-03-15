import pandas as pd

from abc import ABC, abstractmethod


class SensitivityModel(ABC):

    def __init__(self, name: str):
        self.name = name
        self.data = None

    @staticmethod
    @abstractmethod
    def sensitivity_measure(data: pd.DataFrame):
        """
        Computes the sensitivity measure of a sensitivity model based on given data.
        This metric will be an approximation based on available data.
        The data needs to follow a specific naming convention. Each column containing an unobserved confounder needs
        to have the name "Ui" where i indicates the enumeration of the confounder. Similarly with confounders follow
        convention "Xi", treatments "Ti" (yes the treatment might be a vector) and outcome "Yi". Each of these columns
        needs to be discrete / categorical, as the models will work with only given entries.

        :param data: Data used to compute the sensitivity metric.
        :return: Metric found based on data.
        """
        return NotImplemented

    def load_data(self, data: pd.DataFrame):
        self.data = data

    @abstractmethod
    def find_bound(self, sensitivity_measure: float, delta: float, data_uncertainty: bool):
        return NotImplemented
