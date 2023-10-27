# tensor.py
import numpy as np 
from .lazy import LazyBuffer

class Tensor:
    def __init__(self, data=None, requires_grad=False, _lazy_buffer=None):
        self._lazy_buffer = _lazy_buffer or LazyBuffer(data=data)
        self.requires_grad = requires_grad
        self.grad = None 
        self.grad_fn = None

    @property
    def data(self):
        return self._lazy_buffer.realize().data
    
    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        new_buffer = LazyBuffer(op="add", parents=[self._lazy_buffer, other._lazy_buffer])
        result = Tensor(None, requires_grad=self.requires_grad or other.requires_grad, _lazy_buffer=new_buffer)
        result.grad_fn = 'addition'
        return result

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other)
        new_buffer = LazyBuffer(op="sub", parents=[self._lazy_buffer, other._lazy_buffer])
        return Tensor(_lazy_buffer=new_buffer, requires_grad=self.requires_grad or other.requires_grad)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other)
        new_buffer = LazyBuffer(op="mul", parents=[self._lazy_buffer, other._lazy_buffer])
        return Tensor(_lazy_buffer=new_buffer, requires_grad=self.requires_grad or other.requires_grad)

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other)
        new_buffer = LazyBuffer(op="div", parents=[self._lazy_buffer, other._lazy_buffer])
        return Tensor(_lazy_buffer=new_buffer, requires_grad=self.requires_grad or other.requires_grad)

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other)
        new_buffer = LazyBuffer(op="matmul", parents=[self._lazy_buffer, other._lazy_buffer])
        return Tensor(_lazy_buffer=new_buffer, requires_grad=self.requires_grad or other.requires_grad)

    def transpose(self):
        new_buffer = LazyBuffer(op="transpose", parents=[self._lazy_buffer])
        return Tensor(_lazy_buffer=new_buffer, requires_grad=self.requires_grad)

    def sum(self, axis=None):
        # NOTE: This operation doesn't retain laziness for now since numpy's sum is used
        return Tensor(np.sum(self.data, axis=axis), requires_grad=self.requires_grad)
    
    def mean(self, axis=None):
        # NOTE: This operation doesn't retain laziness for now since numpy's mean is used
        return Tensor(np.mean(self.data, axis=axis), requires_grad=self.requires_grad)
    
    def relu(self):
        new_buffer = LazyBuffer(op="relu", parents=[self._lazy_buffer])
        return Tensor(None, requires_grad=self.requires_grad, _lazy_buffer=new_buffer)

    def sigmoid(self):
        new_buffer = LazyBuffer(op="sigmoid", parents=[self._lazy_buffer])
        return Tensor(None, requires_grad=self.requires_grad, _lazy_buffer=new_buffer)