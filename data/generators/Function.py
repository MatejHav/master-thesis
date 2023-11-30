from Abstract import *

class Function:

    def __init__(self, func: Callable):
        self.f = func

    def __call__(self, generator: Generator):
        return self.f(*list(generator))

    def __str__(self):
        return "CustomFunction()"