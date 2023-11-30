from typing import *
import numpy as np

class Value:
    def __init__(self):
        self.value = None

    def get_value(self):
        return self.value

    def __call__(self):
        return self.get_value()

    def __str__(self) -> bool:
        return f"X({self.value})"

    def __eq__(self, other) -> bool:
        if not type(other) == type(self):
            return False
        return self.value == other.value