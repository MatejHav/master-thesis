import numpy as np

from data.generators.Function import Function
from data.generators.Variable import Variable

if __name__ == '__main__':
    F1 = Function(lambda: 5 + 4 * np.random.randn())
    X1 = Variable(F1)
    F2 = Function(lambda: -10 + 7 * np.random.randn())
    X2 = Variable(F2)
    F3 = Function(lambda x1, x2:  3 * x1 - 1 * x2 + 0.5 * np.random.randn())
    X3 = Variable(F3, X1, X2)
    F4 = Function(lambda x1, x3: x1 / x3)
    X4 = Variable(F4, X1, X3)
    print(X4())