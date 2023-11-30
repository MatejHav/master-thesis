from data.generators.Function import Function
from data.generators.Abstract import Value


class Variable(Value):
    def __init__(self, f: Function, *parents):
        super().__init__()
        self.parents = parents
        self.evaluated = False
        self.f = f

    def reset(self):
        if not self.evaluated:
            print("Trying to reset a variable that was not evaluated!")
            return
        self.evaluated = False
        for parent in self.parents:
            parent.reset()

    def get_value(self):
        if not self.evaluated:
            self.value = self.f(map(lambda x: x.get_value(), self.parents))
        return self.value

    def __str__(self):
        return f"Var({self.f}, {', '.join(map(str, self.parents))})"