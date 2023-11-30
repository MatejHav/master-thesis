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
    # States
    Xs1 = Constant(5, is_continuous=False)
    Xs2 = Constant(0, is_continuous=True, range=(0, 2), generator=lambda: np.random.rand() * 2)
    S1 = State([Xs1, Xs2], 0)
    Xs3 = Constant(4, is_continuous=False)
    Xs4 = Constant(0, is_continuous=True, range=(1, 2), generator=lambda: 1 + np.random.rand())
    S2 = State([Xs3, Xs4], 10)
    Xs5 = Constant(6, is_continuous=False)
    Xs6 = Constant(0, is_continuous=True, range=(0, 1), generator=lambda: np.random.rand())
    S3 = State([Xs5, Xs6], 1)
    Xs7 = Constant(0, is_continuous=False)
    Xs8 = Constant(0, is_continuous=True, range=(0, 1), generator=lambda: np.random.rand())
    S4 = State([Xs7, Xs8], -1)
    # Actions
    L = Action([Constant("L", is_continuous=False)])
    R = Action([Constant("R", is_continuous=False)])
    # MDP
    mdp = MDP("Test MDP")
    mdp.add_states([S1, S2, S3])
    mdp.add_transition(S1, L, S2, 1)
    mdp.add_transition(S2, L, S2, 1)
    mdp.add_transition(S1, R, S3, 0.5)
    mdp.add_transition(S1, R, S1, 0.25)
    mdp.add_transition(S1, R, S4, 0.25)
    # Test
    print(mdp.perform_action(S1, L))
    print(mdp.perform_action(S1, R))
    print(mdp.perform_actions(S1, [R]))
    print(mdp.perform_actions(S1, [L, L, L, L]))

