import torch

from models.Agent import *


class QLearningAgent(Agent):
    def __init__(self, name: str, alpha: float, gamma: float, state_size: int, actions: List[Action], hidden_dim: int,
                 num_layers: int):
        super().__init__(name)
        self.alpha = alpha
        self.gamma = gamma
        self.q = torch.nn.Sequential(torch.nn.Linear(state_size, hidden_dim),
                                     *[torch.nn.Sequential(
                                         torch.nn.Linear(hidden_dim, hidden_dim),
                                         torch.nn.ReLU()) for _ in range(num_layers)],
                                     torch.nn.Linear(hidden_dim, len(actions))
                                     )
        self.loss = torch.nn.MSELoss()
        self.actions = [Tensor([a() for a in action.a]) for action in actions]
        self.optimizer = torch.optim.Adam(self.q.parameters())

    def choose(self, state: Tensor) -> int:
        return self.q(state).argmax().item()

    def learn(self, state: Tensor, action: Tensor, next_state: Tensor, reward: float, terminal: bool):
        action_id = self.actions.index(action)
        next_q_value = torch.zeros_like(self.q(next_state)) if terminal else self.q(next_state)
        loss = self.loss(reward + self.gamma * next_q_value.max(), self.q(state)[action_id])
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
