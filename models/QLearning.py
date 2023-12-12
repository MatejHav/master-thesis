import torch

from models.Agent import *


class QLearningAgent(Agent):
    def __init__(self, name: str, alpha: float, gamma: float, state_size: int, action_size: int, hidden_dim: int,
                 num_layers: int):
        super().__init__(name)
        self.alpha = alpha
        self.gamma = gamma
        self.q = torch.nn.Sequential(torch.nn.Linear(state_size, hidden_dim),
                                     *[torch.nn.Sequential(
                                         torch.nn.Linear(hidden_dim, hidden_dim),
                                         torch.nn.ReLU()) for _ in range(num_layers)],
                                     torch.nn.Linear(hidden_dim, action_size)
                                     )
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q.parameters())

    def choose(self, state: Tensor):
        return self.q(state).argmax()

    def learn(self, state: Tensor, action: Tensor, next_state: Tensor, reward: float, terminal: bool):
        temp = self.q(next_state)
        next_q_value = torch.zeros(*temp.shape) if terminal else self.q(next_state)
        loss = self.loss(reward + self.gamma * next_q_value, self.q(state))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
