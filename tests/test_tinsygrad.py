import unittest
import numpy as np
from src.lazy import LazyBuffer
from src.tensor import Tensor

class TestLazyBuffer(unittest.TestCase):

    def test_lazy_addition(self):
        a = LazyBuffer(data=np.array([1, 2, 3]))
        b = LazyBuffer(data=np.array([4, 5, 6]))
        c = LazyBuffer(op="add", parents=[a, b])
        self.assertTrue(np.array_equal(c.realize().data, np.array([5, 7, 9])))

    def test_lazy_subtraction(self):
        a = LazyBuffer(data=np.array([1, 2, 3]))
        b = LazyBuffer(data=np.array([4, 5, 6]))
        c = LazyBuffer(op="sub", parents=[a, b])
        self.assertTrue(np.array_equal(c.realize().data, np.array([-3, -3, -3])))

    def test_lazy_multiplication(self):
        a = LazyBuffer(data=np.array([1, 2, 3]))
        b = LazyBuffer(data=np.array([4, 5, 6]))
        c = LazyBuffer(op="mul", parents=[a, b])
        self.assertTrue(np.array_equal(c.realize().data, np.array([4, 10, 18])))

    # Add more lazy tests as needed

class TestTensor(unittest.TestCase):

    def test_tensor_addition(self):
        a = Tensor(np.array([1, 2, 3]))
        b = Tensor(np.array([4, 5, 6]))
        c = a + b
        self.assertTrue(np.array_equal(c.data, np.array([5, 7, 9])))

    def test_tensor_multiplication(self):
        a = Tensor(np.array([1, 2, 3]))
        b = Tensor(np.array([4, 5, 6]))
        c = a * b
        self.assertTrue(np.array_equal(c.data, np.array([4, 10, 18])))

    def test_tensor_transpose(self):
        a = Tensor(np.array([[1, 2], [3, 4]]))
        b = a.transpose()
        self.assertTrue(np.array_equal(b.data, np.array([[1, 3], [2, 4]])))

    def test_tensor_relu(self):
        a = Tensor(np.array([-1, 0, 1]))
        b = a.relu()
        self.assertTrue(np.array_equal(b.data, np.array([0, 0, 1])))

    # Add more tensor tests as needed

if __name__ == '__main__':
    unittest.main()
