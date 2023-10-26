import numpy as np

class LazyBuffer:
    def __init__(self, data=None, op=None, parents=None):
        self.data = data
        self.op = op
        self.parents = parents or []

    def realize(self):
        """Compute the actual value for this buffer if it hasn't been computed yet."""
        if self.data is None:
            if self.op == "add":
                self.data = self.parents[0].realize().data + self.parents[1].realize().data
            elif self.op == "mul":
                self.data = self.parents[0].realize().data * self.parents[1].realize().data
            else:
                raise ValueError(f"Unknown operation {self.op}")
        return self

    def __repr__(self):
        return f"LazyBuffer(data={self.data}, op={self.op})" 
    

# Test LazyBuffer
a = LazyBuffer(data=np.array([1, 2, 3]))
b = LazyBuffer(data=np.array([4, 5, 6]))
c = LazyBuffer(op="add", parents=[a, b])

c.realize(), print(c.data)
