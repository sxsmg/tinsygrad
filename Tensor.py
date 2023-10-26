import numpy as np 

class Tensor:
    def __init__(self, data, requires_grad=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data 
        self.requires_grad = requires_grad
        self.grad = None 
        self.grad_fn = None

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        result = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        result.grad_fn = 'addition'
        return result

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        result = Tensor(self.data * other.data, requires_grad=(self.requires_grad or other.requires_grad))
        result.grad_fn = "element-wise multiplication"
        return result 

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        result_data = np.matmul(self.data, other.data)
        result = Tensor(result_data, requires_grad=(self.requires_grad or other.requires_grad))
        result.grad_fn = "matrix multiplication"
        return result
