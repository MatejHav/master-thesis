import numpy as np

from data.generators import *


if __name__ == '__main__':
    # Human
    F1 = Function(lambda: 5 + 4 * np.random.randn())
    X1 = Variable(F1)
    F2 = Function(lambda: -10 + 7 * np.random.randn())
    X2 = Variable(F2)
    F3 = Function(lambda x1, x2:  3 * x1 - 1 * x2 + 0.5 * np.random.randn())
    X3 = Variable(F3, X1, X2)
    F4 = Function(lambda x1, x3: x1 / x3)
    X4 = Variable(F4, X1, X3)
    human_features = [X1, X2, X3, X4]
    # Actions
    L = Action([Constant("L", is_continuous=False)])
    R = Action([Constant("R", is_continuous=False)])
    # States
    Xs1 = Constant(5, is_continuous=False)
    Xs2 = Constant(0, is_continuous=True, range=(0, 2), generator=lambda: np.random.rand() * 2)
    S1 = State(lambda _: [Xs1, Xs2], 0)
    Xs3 = Constant(4, is_continuous=False)
    Xs4 = Constant(0, is_continuous=True, range=(1, 2), generator=lambda: 1 + np.random.rand())
    S2 = State(lambda _: [Xs3, Xs4], 10)
    Xs5 = Constant(6, is_continuous=False)
    Xs6 = Constant(0, is_continuous=True, range=(0, 1), generator=lambda: np.random.rand())
    S3 = State(lambda _: [Xs5, Xs6], 1)
    Xs7 = Constant(0, is_continuous=False)
    Xs8 = Constant(0, is_continuous=True, range=(0, 1), generator=lambda: np.random.rand())
    S4 = State(lambda _: [Xs7, Xs8], -1)
    # MDP
    mdp = MDP("Test MDP", 2, 1)
    mdp.add_states([S1, S2, S3, S4])
    mdp.add_transition(S1, L, S2, lambda _: 1)
    mdp.add_transition(S2, L, S2, lambda _: 1)
    mdp.add_transition(S1, R, S3, lambda _: 0.5)
    mdp.add_transition(S1, R, S1, lambda _: 0.25)
    mdp.add_transition(S1, R, S4, lambda _: 0.25)
    mdp.add_transition(S2, R, S2, lambda _: 1)
    mdp.add_transition(S3, L, S3, lambda _: 1)
    mdp.add_transition(S3, R, S3, lambda _: 1)
    mdp.add_transition(S4, L, S4, lambda _: 1)
    mdp.add_transition(S4, R, S4, lambda _: 1)
    # Generator
    generator = Generator(mdp, human_features)
    table = generator.generate_uniform_data(10000, 50, lambda _: S1, 50)
    print(table)



