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


class TestTensor(unittest.TestCase):

    def test_tensor_addition(self):
        a = Tensor(np.array([1, 2, 3]))
        b = Tensor(np.array([4, 5, 6]))
        c = a + b
        self.assertTrue(np.array_equal(c.data, np.array([5, 7, 9])))

# You can add more tests or classes as needed.

if __name__ == '__main__':
    unittest.main()
