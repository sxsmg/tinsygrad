#mnist_cnn.py
import numpy as np
import gzip
import struct
from src import Tensor,NeuralNet, Linear, ReLU, Dropout,MSELoss, SGD, Adam, CrossEntropyLoss, Conv2D, Flatten

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        # Read and discard the magic number
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read all data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape data into 3D tensor (num, rows, cols)
        data = data.reshape(num, rows, cols)
        # Flatten and normalize
        return data.reshape(num, -1) / 255.0

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        # Read and discard the magic number and number of labels
        magic, num = struct.unpack(">II", f.read(8))
        # Read all data
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data

# Load training and test datasets
x_train = load_mnist_images('./tests/mnist/train-images-idx3-ubyte.gz')
y_train = load_mnist_labels('./tests/mnist/train-labels-idx1-ubyte.gz')
x_test = load_mnist_images('./tests/mnist/t10k-images-idx3-ubyte.gz')
y_test = load_mnist_labels('./tests/mnist/t10k-labels-idx1-ubyte.gz')

# One-hot encode the labels
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

y_train_one_hot = one_hot_encode(y_train)
y_test_one_hot = one_hot_encode(y_test)

x_train = x_train.reshape(-1, 1, 28, 28) #for CNN
x_test = x_test.reshape(-1, 1, 28, 28)   #for CNN

model = NeuralNet([
    Conv2D(1, 32, kernel_size=5, padding=2),
    ReLU(),
    Conv2D(32, 64, kernel_size=5, padding=2),
    ReLU(),
    Flatten(),
    Linear(64 * 28 * 28, 128),
    ReLU(),
    Linear(128, 10)
])


# Hyperparameters
learning_rate = 0.001
epochs = 10
batch_size = 64

# Loss and Optimizer
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    # Shuffle your data or use a method to get mini-batches
    for batch in range(0, len(x_train), batch_size):
        x_batch = Tensor(x_train[batch:batch+batch_size])
        y_batch = Tensor(y_train_one_hot[batch:batch+batch_size])
        
        # Forward pass
        outputs = model.forward(x_batch)
        loss = criterion.forward(outputs, y_batch)

        # Backward pass and update
        model.backward(loss)
        optimizer.step()

    test_outputs = model.forward(Tensor(x_test))
    predicted_classes = np.argmax(test_outputs.data, axis=1)
    accuracy = np.mean(predicted_classes == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
