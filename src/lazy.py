#lazy.py
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
            elif self.op == "sub":
                self.data = self.parents[0].realize().data - self.parents[1].realize().data
            elif self.op == "mul":
                self.data = self.parents[0].realize().data * self.parents[1].realize().data
            elif self.op == "div":
                self.data = self.parents[0].realize().data / self.parents[1].realize().data
            elif self.op == "matmul":
                self.data = np.matmul(self.parents[0].realize().data, self.parents[1].realize().data)
            elif self.op == "transpose":
                self.data = np.transpose(self.parents[0].realize().data)
            elif self.op == "relu":
                self.data = np.maximum(0, self.parents[0].realize().data)
            else:
                raise ValueError(f"Unknown operation {self.op}")
        return self


    def __repr__(self):
        return f"LazyBuffer(data={self.data}, op={self.op})" 
    
