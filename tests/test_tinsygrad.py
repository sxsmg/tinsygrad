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

    def test_lazy_division(self):
        a = LazyBuffer(data=np.array([1, 2, 6]))
        b = LazyBuffer(data=np.array([1, 2, 3]))
        c = LazyBuffer(op="div", parents=[a, b])
        self.assertTrue(np.array_equal(c.realize().data, np.array([1, 1, 2])))

    def test_lazy_matrix_multiplication(self):
        a = LazyBuffer(data=np.array([[1, 2], [3, 4]]))
        b = LazyBuffer(data=np.array([[2, 0], [0, 2]]))
        c = LazyBuffer(op="matmul", parents=[a, b])
        self.assertTrue(np.array_equal(c.realize().data, np.array([[2, 4], [6, 8]])))

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

    def test_tensor_mean(self):
        a = Tensor(np.array([1, 2, 3, 4]))
        b = a.mean()
        self.assertEqual(b.data, 2.5)

    def test_tensor_sigmoid(self):
        a = Tensor(np.array([-1, 0, 1]))
        b = a.sigmoid()
        expected = 1 / (1 + np.exp(-a.data))
        self.assertTrue(np.allclose(b.data, expected))

    def test_tensor_sum(self):
        a = Tensor(np.array([[1, 2], [3, 4]]))
        b = a.sum()
        self.assertEqual(b.data, 10)

    def test_tensor_chain_operations(self):
            a = Tensor(np.array([1, -2, 3]))
            b = Tensor(np.array([2, 3, -4]))
            c = (a + b).relu().sigmoid()
            intermediate_result = a.data + b.data
            intermediate_result[intermediate_result < 0] = 0  # ReLU operation
            expected = 1 / (1 + np.exp(-intermediate_result))  # Sigmoid operation
            self.assertTrue(np.allclose(c.data, expected))



if __name__ == '__main__':
    unittest.main()
