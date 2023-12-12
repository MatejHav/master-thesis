import numpy as np

from data.generators import *
from data.builders import *

def build_maze(m, n, default_r, max_r, p):
    # MDP
    # Actions
    mdp_builder = (MDPBuilder("maze_mdp", 3, 1)
                   .add_action("L", [Constant("L", is_continuous=False)])
                   .add_action("D", [Constant("D", is_continuous=False)])
                   .add_action("U", [Constant("U", is_continuous=False)])
                   .add_action("R", [Constant("R", is_continuous=False)]))
    # States
    mdp_builder.add_constant_discrete_state("S(0, 0)", [0, 0, False], default_r)
    for x in range(1, m):
        is_wall = False
        if x > 1 and np.random.rand() < p:
            is_wall = True
        mdp_builder.add_constant_discrete_state(f"S({x}, 0)", [x, 0, is_wall], default_r)
        if not is_wall:
            if not mdp_builder.get_state(f"S({x-1}, 0)").X(None)[-1]():
                mdp_builder.connect_states(f"S({x-1}, 0)", f"S({x}, 0)", "R", lambda human: 1 - human[0]())
                mdp_builder.connect_states(f"S({x}, 0)", f"S({x-1}, 0)", "L", lambda human: 1 - human[0]())
                mdp_builder.connect_states(f"S({x-1}, 0)", f"S({x-1}, 0)", "R", lambda human: human[0]())
                mdp_builder.connect_states(f"S({x}, 0)", f"S({x}, 0)", "L", lambda human: human[0]())
            else:
                mdp_builder.connect_states(f"S({x-1}, 0)", f"S({x-1}, 0)", "R", lambda human: 1)
                mdp_builder.connect_states(f"S({x}, 0)", f"S({x}, 0)", "L", lambda human: 1)
    for y in range(1, n):
        is_wall = False
        if y > 1 and np.random.rand() < p:
            is_wall = True
        mdp_builder.add_constant_discrete_state(f"S(0, {y})", [0, y, is_wall], default_r)
        if not is_wall:
            if not mdp_builder.get_state(f"S(0, {y-1})").X(None)[-1]():
                mdp_builder.connect_states(f"S(0, {y-1})", f"S(0, {y})", "U", lambda human: 1 - human[0]())
                mdp_builder.connect_states(f"S(0, {y})", f"S(0, {y-1})", "D", lambda human: 1 - human[0]())
                mdp_builder.connect_states(f"S(0, {y - 1})", f"S(0, {y-1})", "U", lambda human: human[0]())
                mdp_builder.connect_states(f"S(0, {y})", f"S(0, {y})", "D", lambda human: human[0]())
            else:
                mdp_builder.connect_states(f"S(0, {y-1})", f"S(0, {y-1})", "U", lambda human: 1)
                mdp_builder.connect_states(f"S(0, {y})", f"S(0, {y})", "D", lambda human: 1)
    for x in range(1, m):
        for y in range(1, n):
            r = default_r
            is_wall = False
            if x != 0 and np.random.rand() < p:
                is_wall = True
            if x == m - 1 and y == n - 1:
                is_wall = False
                r = max_r
            mdp_builder.add_constant_discrete_state(f"S({x}, {y})", [x, y, is_wall], r)
            if not is_wall:
                if not mdp_builder.get_state(f"S({x - 1}, {y})").X(None)[-1]():
                    mdp_builder.connect_states(f"S({x - 1}, {y})", f"S({x}, {y})", "R", lambda human: 1 - human[0]())
                    mdp_builder.connect_states(f"S({x}, {y})", f"S({x - 1}, {y})", "L", lambda human: 1 - human[0]())
                    mdp_builder.connect_states(f"S({x - 1}, {y})", f"S({x - 1}, {y})", "R", lambda human: human[0]())
                    mdp_builder.connect_states(f"S({x}, {y})", f"S({x}, {y})", "L", lambda human: human[0]())
                else:
                    mdp_builder.connect_states(f"S({x - 1}, {y})", f"S({x - 1}, {y})", "R", lambda human: 1)
                    mdp_builder.connect_states(f"S({x}, {y})", f"S({x}, {y})", "L", lambda human: 1)
                if not mdp_builder.get_state(f"S({x}, {y - 1})").X(None)[-1]():
                    mdp_builder.connect_states(f"S({x}, {y - 1})", f"S({x}, {y})", "U", lambda human: 1 - human[0]())
                    mdp_builder.connect_states(f"S({x}, {y})", f"S({x}, {y - 1})", "D", lambda human: 1 - human[0]())
                    mdp_builder.connect_states(f"S({x}, {y-1})", f"S({x}, {y-1})", "U", lambda human: human[0]())
                    mdp_builder.connect_states(f"S({x}, {y})", f"S({x}, {y})", "D", lambda human: human[0]())
                else:
                    mdp_builder.connect_states(f"S({x}, {y - 1})", f"S({x}, {y - 1})", "U", lambda human: 1)
                    mdp_builder.connect_states(f"S({x}, {y})", f"S({x}, {y})", "D", lambda human: 1)
    mdp = mdp_builder.build()
    return mdp_builder, mdp


if __name__ == '__main__':
    m, n = 20, 40
    default_r = -0.01
    max_r = 10
    p = 0.2
    mdp_builder, mdp = build_maze(m, n, default_r, max_r, p)
    # Human
    F1 = Function(lambda: np.random.rand() / 4)
    X1 = Variable(F1)
    human_features = [X1]
    # Generator
    generator = Generator(mdp, human_features)
    df = generator.generate_uniform_data(num_of_rows=10000, n_jobs=50, starting_state=lambda _: mdp_builder.get_state("S(0, 0)"), max_iter=100, verbose=1)
    df.to_csv("maze_data.csv")




