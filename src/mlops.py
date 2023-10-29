#mlops.py 
from .tensor import Tensor
import numpy as np
from numba import jit
from .lazy import LazyBuffer

#layers
class Linear:
    def __init__(self, in_features, out_features):
        # Initialize with a lazy buffer for weights and biases
        self.weights = Tensor(_lazy_buffer=LazyBuffer(data=np.random.randn(in_features, out_features)))
        self.bias = Tensor(_lazy_buffer=LazyBuffer(data=np.random.randn(out_features)))
    
    def forward(self, x):
        return x @ self.weights + self.bias

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        # Initializing with lazy buffers
        self.weights = Tensor(_lazy_buffer=LazyBuffer(data=np.random.randn(out_channels, in_channels, kernel_size, kernel_size)))
        self.bias = Tensor(_lazy_buffer=LazyBuffer(data=np.random.randn(out_channels)))
        self.stride = stride
        self.padding = padding

    
    def _im2col(self, input_data, kernel_height, kernel_width, stride):
        N, C, H, W = input_data.shape
        out_h = (H + 2*self.padding - kernel_height) // stride + 1
        out_w = (W + 2*self.padding - kernel_width) // stride + 1

        img = np.pad(input_data, [(0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)], 'constant')
        
        # Create strided version of input. Every stride creates a channel
        shape = (N, C, out_h, out_w, kernel_height, kernel_width)
        strides = (*img.strides[:-2], img.strides[-2] * stride, img.strides[-1] * stride, *img.strides[-2:])
        
        windows = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
        print("working")
        # Reshape the windows into 2D array
        cols = windows.reshape(N*out_h*out_w, -1)
        return cols



    def forward(self, x):
        # Lazy evaluation
        x_data = x._lazy_buffer.realize().data
        weights_data = self.weights._lazy_buffer.realize().data

        # Extract shape information
        batch_size, in_channels, H, W = x_data.shape
        out_channels, _, kernel_height, kernel_width = weights_data.shape

        # Calculate the output dimensions
        H_out = (H + 2 * self.padding - kernel_height) // self.stride + 1
        W_out = (W + 2 * self.padding - kernel_width) // self.stride + 1

        # Convert tensors to 2D matrices suitable for matmul
        cols = self._im2col(x_data, kernel_height, kernel_width, self.stride)
        weights_2d = weights_data.reshape(out_channels, -1)
        
        out = np.matmul(weights_2d, cols.T) + self.bias.data.reshape(-1, 1)
        out = out.reshape(out_channels, H_out, W_out, batch_size).transpose(3, 0, 1, 2)
        assert out.shape == (batch_size, out_channels, H_out, W_out), f"Expected output shape: {(batch_size, out_channels, H_out, W_out)}, but got: {out.shape}"
        return Tensor(out)



#layer ops
class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.train_mode = True

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

    def forward(self, x):
        if not self.train_mode:
            return x
        
        self.mask = (np.random.rand(*x.data.shape) > self.p) / (1.0 - self.p)
        return x * self.mask

class Flatten:
    def forward(self, x):
        batch_size = x.data.shape[0]
        return Tensor(x.data.reshape(batch_size, -1))

#activation functions
class ReLU:
    def forward(self, x):
        return x.relu()

class Sigmoid:
    def forward(self, x):
        return x.sigmoid()

class Tanh:
    def forward(self, x):
        return np.tanh(x.data)

#loss functions
class MSELoss:
    def forward(self, predictions, labels):
        return ((predictions - labels) ** 2).mean()

class CrossEntropyLoss:
    def forward(self, predictions, labels):
        # Subtracting the max for numerical stability
        logits = predictions.data - np.max(predictions.data, axis=1, keepdims=True)
        
        self.softmax = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        self.labels = labels
        return -np.sum(labels.data * np.log(self.softmax + 1e-10)) / labels.data.shape[0]

#optimizers
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        for param in self.parameters:
            if param.grad is not None:  # Check if gradient has been computed
                param.data -= self.lr * param.grad

@jit(nopython=True)
def adam_update(m, v, g, t, betas, lr, eps):
    m = betas[0] * m + (1 - betas[0]) * g
    v = betas[1] * v + (1 - betas[1]) * g**2
    m_hat = m / (1 - betas[0]**t)
    v_hat = v / (1 - betas[1]**t)
    param_update = lr * m_hat / (np.sqrt(v_hat) + eps)
    return m, v, param_update
    
class Adam:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0
    
    def step(self):
        self.t += 1
        for idx, param in enumerate(self.parameters):
            g = param.grad
            if g is None:
                continue  # skip this parameter or set g = 0.0

            self.m[idx], self.v[idx], param_update = adam_update(
                self.m[idx], self.v[idx], g, self.t, self.betas, self.lr, self.eps
            )

            param.data -= param_update

#nn container
class NeuralNet:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                grad = layer.backward(grad)

    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                params.append(layer.weights)
            if hasattr(layer, 'bias'):
                params.append(layer.bias)
        return params


#devices
class Device:
    def __init__(self, type='cpu'):
        self.type = type
    
    def move_to_device(self, tensor):
        if self.type == 'gpu':
            # Pseudo code. In reality, you would use a library like CuPy here.
            tensor.data = move_data_to_gpu(tensor.data)
        return tensor
    
    def __repr__(self):
        return f"Device(type='{self.type}')"
