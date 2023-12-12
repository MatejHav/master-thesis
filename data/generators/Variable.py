from data.generators.Function import Function
from data.generators.Abstract import Value


class Variable(Value):
    def __init__(self, f: Function, *parents):
        super().__init__()
        self.parents = parents
        self.evaluated = False
        self.f = f

    def reset(self):
        self.evaluated = False
        for parent in self.parents:
            parent.reset()

    def get_value(self):
        if not self.evaluated:
            self.value = self.f(list(map(lambda x: x.get_value(), self.parents)))
            self.evaluated = True
        return self.value

    def __str__(self):
        return f"Var({self.f}, {', '.join(map(str, self.parents))})"