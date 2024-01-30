from models.Agent import *


class CustomAgent(Agent):
    def __init__(self, name: str, pick_probability: Callable[[Tensor], List[float]]):
        super().__init__(name)
        self.pi = pick_probability

    def choose(self, state: Tensor) -> int:
        return np.argmax(self.pi(state))
    def policy(self, state: Tensor) -> List[float]:
        return self.pi(state)

    def learn(self, state: Tensor, action: Tensor, next_state: Tensor, reward: float, terminal: bool):
        return 0
