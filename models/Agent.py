from data.generators import *


class Agent:

    def __init__(self, name: str):
        self.name = name

    def choose(self, state: Tensor) -> int:
        pass

    def policy(self, state: Tensor) -> List[float]:
        pass

    def learn(self, state: Tensor, action: Tensor, next_state: Tensor, reward: float, terminal: bool):
        pass
