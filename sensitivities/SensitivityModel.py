import pandas as pd

from abc import ABC, abstractmethod


class SensitivityModel(ABC):

    def __init__(self, name: str, data_minimum: int):
        self.name = name
        self.data_min = data_minimum
        self.data = None

    @staticmethod
    @abstractmethod
    def sensitivity_measure(data: pd.DataFrame):
        return NotImplemented

    def load_data(self, data: pd.DataFrame):
        self.data = data

    @abstractmethod
    def find_bound(self, sensitivity_measure: float, delta: float, data_uncertainty: bool):
        return NotImplemented
