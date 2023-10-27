# tensor.py
import numpy as np 
from .lazy import LazyBuffer

class Tensor:
    def __init__(self, data=None, requires_grad=False, _lazy_buffer=None):
        if data is not None and not isinstance(data, np.ndarray):
            data = np.array(data)
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
        result.grad_fn = 'add'
        return result

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other)
        new_buffer = LazyBuffer(op="sub", parents=[self._lazy_buffer, other._lazy_buffer])
        result = Tensor(_lazy_buffer=new_buffer, requires_grad=self.requires_grad or other.requires_grad)
        result.grad_fn = "sub"
        return result

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other)
        new_buffer = LazyBuffer(op="mul", parents=[self._lazy_buffer, other._lazy_buffer])
        result = Tensor(_lazy_buffer=new_buffer, requires_grad=self.requires_grad or other.requires_grad)
        result.grad_fn = "mul" 
        return result
        
    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other)
        new_buffer = LazyBuffer(op="div", parents=[self._lazy_buffer, other._lazy_buffer])
        result = Tensor(_lazy_buffer=new_buffer, requires_grad=self.requires_grad or other.requires_grad)
        result.grad_fn = "div"
        return result

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(data=other)
        new_buffer = LazyBuffer(op="matmul", parents=[self._lazy_buffer, other._lazy_buffer])
        result = Tensor(_lazy_buffer=new_buffer, requires_grad=self.requires_grad or other.requires_grad)
        result.grad_fn = "matmul"
        return result

    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        new_buffer = LazyBuffer(op="power", parents=[self._lazy_buffer, other._lazy_buffer])
        return Tensor(_lazy_buffer=new_buffer, requires_grad=self.requires_grad or other.requires_grad)

    def transpose(self):
        new_buffer = LazyBuffer(op="transpose", parents=[self._lazy_buffer])
        return Tensor(_lazy_buffer=new_buffer, requires_grad=self.requires_grad)

    def sum(self, axis=None):
        new_buffer = LazyBuffer(op="sum", parents=[self._lazy_buffer])
        return Tensor(_lazy_buffer=new_buffer, requires_grad=self.requires_grad)
    
    def mean(self, axis=None):
        # NOTE: This operation doesn't retain laziness for now since numpy's mean is used
        return Tensor(np.mean(self.data, axis=axis), requires_grad=self.requires_grad)
    
    def relu(self):
        new_buffer = LazyBuffer(op="relu", parents=[self._lazy_buffer])
        return Tensor(None, requires_grad=self.requires_grad, _lazy_buffer=new_buffer)

    def sigmoid(self):
        new_buffer = LazyBuffer(op="sigmoid", parents=[self._lazy_buffer])
        return Tensor(None, requires_grad=self.requires_grad, _lazy_buffer=new_buffer)

    def backward(self, grad=None):
        if not self.requires_grad:
            return 
        if grad is None:
            grad = np.ones_like(self.data)

        if self.grad_fn is None:
            return

        if self.grad_fn == "add":
            a, b = self.parents
            a.backward(grad)
            b.backward(grad)

        elif self.grad_fn == "sub":
            a, b = self.parents
            a.backward(grad)
            b.backward(-grad)

        elif self.grad_fn == "mul":
            a, b = self.parents
            a.backward(grad * b.data)
            b.backward(grad * a.data)

        elif self.grad_fn == "div":
            a, b = self.parents
            a.backward(grad / b.data)
            b.backward(-grad * a.data / (b.data**2))

        elif self.grad_fn == "matmul":
            a, b = self.parents
            a.backward(grad @ b.data.T)
            b.backward(a.data.T @ grad)

        elif self.grad_fn == "relu":
            a, = self.parents
            grad_relu = np.where(a.data > 0, grad, 0)
            a.backward(grad_relu)

        elif self.grad_fn == "sigmoid":
            a, = self.parents
            sigmoid_grad = self.data * (1 - self.data)
            a.backward(grad * sigmoid_grad)

        elif self.grad_fn == "sum":
            a, = self.parents
            a.backward(np.broadcast_to(grad, a.data.shape))

        else:
            raise ValueError(f"Gradient not implemented for {self.grad_fn}")
