from data.generators import *
from data.builders import *


def build_maze(m: int, n: int, default_r: float, max_r: float, p: float) -> Tuple[MDPBuilder, MDP]:
    """
    Builds a basic MDP representing a rectangular maze.
    The starting state is always the up-rightmost point and the terminal reward state is the down-leftmost state.
    The maze also has 1 global confounder (human[0]) which indicates how likely is the human to succeed in their action.
    :param m: Width of the maze
    :param n: Height of the maze
    :param default_r: Reward in each non-terminal state
    :param max_r: Reward in the terminal state
    :param p: Probability of a state being a wall (being non-traversable)
    :return: MDPBuilder and the generated MDP
    """
    assert m > 0 and n > 0, f"Both m and n need to be positive, m={m} and n={n} received instead."
    assert default_r < max_r, f"Default reward must be smaller than maximal reward, obtained default_r={default_r} and max_r={max_r}."
    assert 0 <= p <= 1, f"Probability of wall must be between 0 and 1, got {p} instead."
    # MDP
    # Actions
    mdp_builder = (MDPBuilder("maze_mdp", 3, 1)
                   .add_action("L", [Constant(0, is_continuous=False)])
                   .add_action("D", [Constant(1, is_continuous=False)])
                   .add_action("U", [Constant(2, is_continuous=False)])
                   .add_action("R", [Constant(3, is_continuous=False)]))
    # States
    mdp_builder.add_constant_discrete_state("S(0, 0)", [0, 0, False], default_r)
    mdp_builder.connect_states("S(0, 0)", "S(0, 0)", "D", lambda human: 1)
    mdp_builder.connect_states("S(0, 0)", "S(0, 0)", "L", lambda human: 1)
    for x in range(1, m):
        is_wall = False
        if x > 1 and np.random.rand() < p:
            is_wall = True
        mdp_builder.add_constant_discrete_state(f"S({x}, 0)", [x, 0, is_wall], default_r)
        if not is_wall:
            if not mdp_builder.get_state(f"S({x - 1}, 0)").X[-1]():
                mdp_builder.connect_states(f"S({x - 1}, 0)", f"S({x}, 0)", "R", lambda human: 1 - human[0]())
                mdp_builder.connect_states(f"S({x}, 0)", f"S({x - 1}, 0)", "L", lambda human: 1 - human[0]())
                mdp_builder.connect_states(f"S({x - 1}, 0)", f"S({x - 1}, 0)", "R", lambda human: human[0]())
                mdp_builder.connect_states(f"S({x}, 0)", f"S({x}, 0)", "L", lambda human: human[0]())
            else:
                mdp_builder.connect_states(f"S({x - 1}, 0)", f"S({x - 1}, 0)", "R", lambda human: 1)
                mdp_builder.connect_states(f"S({x}, 0)", f"S({x}, 0)", "L", lambda human: 1)
            mdp_builder.connect_states(f"S({x}, 0)", f"S({x}, 0)", "D", lambda human: 1)
    for y in range(1, n):
        is_wall = False
        if y > 1 and np.random.rand() < p:
            is_wall = True
        mdp_builder.add_constant_discrete_state(f"S(0, {y})", [0, y, is_wall], default_r)
        if not is_wall:
            if not mdp_builder.get_state(f"S(0, {y - 1})").X[-1]():
                mdp_builder.connect_states(f"S(0, {y - 1})", f"S(0, {y})", "U", lambda human: 1 - human[0]())
                mdp_builder.connect_states(f"S(0, {y})", f"S(0, {y - 1})", "D", lambda human: 1 - human[0]())
                mdp_builder.connect_states(f"S(0, {y - 1})", f"S(0, {y - 1})", "U", lambda human: human[0]())
                mdp_builder.connect_states(f"S(0, {y})", f"S(0, {y})", "D", lambda human: human[0]())
            else:
                mdp_builder.connect_states(f"S(0, {y - 1})", f"S(0, {y - 1})", "U", lambda human: 1)
                mdp_builder.connect_states(f"S(0, {y})", f"S(0, {y})", "D", lambda human: 1)
            mdp_builder.connect_states(f"S(0, {y})", f"S(0, {y})", "R", lambda human: 1)
    mdp_builder.connect_states(f"S(0, {n-1})", f"S(0, {n-1})", "U", lambda human: 1)
    mdp_builder.connect_states(f"S(0, {n - 1})", f"S(0, {n - 1})", "L", lambda human: 1)
    mdp_builder.connect_states(f"S({m - 1}, 0)", f"S({m - 1}, 0)", "D", lambda human: 1)
    mdp_builder.connect_states(f"S({m - 1}, 0)", f"S({m - 1}, 0)", "R", lambda human: 1)

    for x in range(1, m):
        for y in range(1, n):
            r = default_r
            is_wall = False
            if 1 < x < m - 1 and 1 < y < n - 1 and np.random.rand() < p:
                is_wall = True
            if x == m - 1 and y == n - 1:
                is_wall = False
                r = max_r
            mdp_builder.add_constant_discrete_state(f"S({x}, {y})", [x, y, is_wall], r,
                                                    terminal=x == m - 1 and y == n - 1)
            if not is_wall:
                if not mdp_builder.get_state(f"S({x - 1}, {y})").X[-1]():
                    mdp_builder.connect_states(f"S({x - 1}, {y})", f"S({x}, {y})", "R", lambda human: 1 - human[0]())
                    if not (x == m - 1 and y == n - 1):
                        mdp_builder.connect_states(f"S({x}, {y})", f"S({x - 1}, {y})", "L",
                                                   lambda human: 1 - human[0]())
                    mdp_builder.connect_states(f"S({x - 1}, {y})", f"S({x - 1}, {y})", "R", lambda human: human[0]())
                    if not (x == m - 1 and y == n - 1):
                        mdp_builder.connect_states(f"S({x}, {y})", f"S({x}, {y})", "L", lambda human: human[0]())
                else:
                    mdp_builder.connect_states(f"S({x - 1}, {y})", f"S({x - 1}, {y})", "R", lambda human: 1)
                    if not (x == m - 1 and y == n - 1):
                        mdp_builder.connect_states(f"S({x}, {y})", f"S({x}, {y})", "L", lambda human: 1)
                if not mdp_builder.get_state(f"S({x}, {y - 1})").X[-1]():
                    mdp_builder.connect_states(f"S({x}, {y - 1})", f"S({x}, {y})", "U", lambda human: 1 - human[0]())
                    if not (x == m - 1 and y == n - 1):
                        mdp_builder.connect_states(f"S({x}, {y})", f"S({x}, {y - 1})", "D", lambda human: 1 - human[0]())
                    mdp_builder.connect_states(f"S({x}, {y - 1})", f"S({x}, {y - 1})", "U", lambda human: human[0]())
                    if not (x == m - 1 and y == n - 1):
                        mdp_builder.connect_states(f"S({x}, {y})", f"S({x}, {y})", "D", lambda human: human[0]())
                else:
                    mdp_builder.connect_states(f"S({x}, {y - 1})", f"S({x}, {y - 1})", "U", lambda human: 1)
                    if not (x == m - 1 and y == n - 1):
                        mdp_builder.connect_states(f"S({x}, {y})", f"S({x}, {y})", "D", lambda human: 1)
                if x == m - 1 and y != n - 1:
                    mdp_builder.connect_states(f"S({x}, {y})", f"S({x}, {y})", "R", lambda human: 1)
                if y == n - 1 and x != m - 1:
                    mdp_builder.connect_states(f"S({x}, {y})", f"S({x}, {y})", "U", lambda human: 1)
    mdp = mdp_builder.build()
    return mdp_builder, mdp

def build_basic_mdp():
    """
    Generates a 2x2 maze with no walls. So basically a 4 state MDP with transitions to neighbouring states.
    There is just 1 confounder, which is human[0] which indicates how likely an agent is to succeed in their action.
    """
    return build_maze(2, 2, default_r=-1, max_r=100, p=0)

def build_4x4_blank_mdp():
    return build_maze(4, 4, default_r=-1, max_r=10, p=0)