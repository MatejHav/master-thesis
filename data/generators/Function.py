from data.generators.Abstract import *

class Function:

    def __init__(self, func: Callable):
        self.f = func

    def __call__(self, values: List):
        return self.f(*values)

    def __str__(self):
        return "CustomFunction()"