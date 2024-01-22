% State space
int: S;
set of int: states = 1..S;
% Action space
int: A;
set of int: actions = 1..A;
% Transition bounds
array[states, actions, states, 1..2] of int: transition_bounds;
% Rewards
array[states, actions, states] of int: rewards;
% Agent policy
array[states, actions] of int: policy;

% Variables
% P(state, action -> next_state)
array[states, actions, states] of var int: P;
% V(state)
array[states] of var int: V;

% Constrain 0: Value function bounds
constraint forall(state in states)(V[state] <= 100 /\ -100 <= V[state]);

% Constraint 1: Transition bounds
% Every transistion P is between 0 and 1, or tighter bound if specified by transitions
constraint forall(state in states, action in actions, next_state in states)(transition_bounds[state, action, next_state, 1] <= P[state, action, next_state] /\ P[state, action, next_state] <= transition_bounds[state, action, next_state, 2]);

% Constraint 2: Transistions with the same state-action pair sum up to 1
constraint forall(state in states, action in actions)(sum([P[state, action, next_state] | next_state in states]) == 100);

% Constraint 3: Value function definition
constraint forall(state in states)(V[state] == sum([policy[state, action] * sum([P[state, action, next_state] * (rewards[state, action, next_state] + V[next_state])| next_state in states]) | action in actions]));

% Optimize by minimizing V(s_0)
solve minimize V[1];

output ["P(\(state), \(action) -> \(next_state)) = " ++ show(P[state, action, next_state] / 100) | state in states, action in actions, next_state in states] ++ [show(V[1] / (100 * 100 * A))];
