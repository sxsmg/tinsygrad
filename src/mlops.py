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

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.weights = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = Tensor(np.random.randn(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Extract shape information
        batch_size, in_channels, H, W = x.data.shape
        out_channels, _, kernel_height, kernel_width = self.weights.data.shape

        # Calculate output dimensions
        H_out = (H - kernel_height + 2 * self.padding) // self.stride + 1
        W_out = (W - kernel_width + 2 * self.padding) // self.stride + 1

        # Initialize output tensor
        out = np.zeros((batch_size, out_channels, H_out, W_out))

        # Apply padding to the input tensor if specified
        if self.padding > 0:
            x_padded = np.pad(x.data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x.data

        # Convolution operation
        for i in range(0, H_out * self.stride, self.stride):
            for j in range(0, W_out * self.stride, self.stride):
                x_slice = x_padded[:, :, i:i+kernel_height, j:j+kernel_width]
                for k in range(out_channels):
                    out[:, k, i//self.stride, j//self.stride] = np.sum(x_slice * self.weights.data[k, :, :, :], axis=(1, 2, 3)) + self.bias.data[k]

        return Tensor(out)

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
