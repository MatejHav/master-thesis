from data.generators.Abstract import Value

class Constant(Value):
    def __init__(self, value, is_continuous: bool, **kwargs):
        super().__init__()
        self.value = value
        self.is_continuous = is_continuous
        if is_continuous:
            try:
                self.range = kwargs['range']
                self.generator = kwargs['generator']
            except:
                print("You have not defined the range for a continuous constant.")

    def __str__(self) -> str:
        if self.is_continuous:
            return f"Const({self.range})"
        return f"Const({self.value})"

    def get_value(self):
        if self.is_continuous:
            self.value = self.generator()
        return self.value

    def __eq__(self, other):
        if not type(other) == type(self):
            return False
        if self.is_continuous and other.is_continuous:
            return self.range == other.range
        return self.value == other.value
