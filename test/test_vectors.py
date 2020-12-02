import random
import unittest

import pytest
import numpy as np
import torch

from extreme_maths.vectors import EMVector
from extreme_maths.vectors import EMVectorCuda
from extreme_maths.vectors import EMVectorOCL
from extreme_maths.vectors import EMVectorThreaded


class TestEMVector(unittest.TestCase):
    cls = EMVector

    def test_add(self):
        arr1 = [random.random() for _ in range(100)]
        arr2 = [random.random() for _ in range(100)]

        np1 = np.array(arr1, dtype=np.float)
        np2 = np.array(arr2, dtype=np.float)
        np_res = np.add(arr1, arr2, dtype=np.float)

        t1 = torch.tensor(arr1)
        t2 = torch.tensor(arr2)
        t_res = torch.add(t1, t2)
        np.testing.assert_allclose(np_res, t_res, rtol=1e-5, atol=0)

        v1 = self.cls(arr1)
        v2 = self.cls(arr2)
        v_res = (v1 + v2).result()

        np.testing.assert_allclose(np_res, v_res, rtol=1e-5, atol=0)

    def test_iadd(self):
        arr1 = [random.random() for _ in range(100)]
        arr2 = [random.random() for _ in range(100)]

        np1 = np.array(arr1, dtype=np.float)
        np2 = np.array(arr2, dtype=np.float)
        np_res = np.add(arr1, arr2, dtype=np.float)

        t1 = torch.tensor(arr1)
        t2 = torch.tensor(arr2)
        t_res = torch.add(t1, t2)
        np.testing.assert_allclose(np_res, t_res, rtol=1e-5, atol=0)

        v1 = self.cls(arr1)
        v2 = self.cls(arr2)
        v1 += v2
        v_res = v1.result()

        np.testing.assert_allclose(np_res, v_res, rtol=1e-5, atol=0)

    def test_sub(self):
        arr1 = [random.random() for _ in range(100)]
        arr2 = [random.random() for _ in range(100)]

        np1 = np.array(arr1, dtype=np.float)
        np2 = np.array(arr2, dtype=np.float)
        np_res = np.subtract(arr1, arr2, dtype=np.float)

        t1 = torch.tensor(arr1)
        t2 = torch.tensor(arr2)
        t_res = torch.sub(t1, t2)
        np.testing.assert_allclose(np_res, t_res, rtol=1e-4, atol=0)

        v1 = self.cls(arr1)
        v2 = self.cls(arr2)
        v_res = (v1 - v2).result()

        np.testing.assert_allclose(np_res, v_res, rtol=1e-4, atol=0)

    def test_isub(self):
        arr1 = [random.random() for _ in range(100)]
        arr2 = [random.random() for _ in range(100)]

        np1 = np.array(arr1, dtype=np.float)
        np2 = np.array(arr2, dtype=np.float)
        np_res = np.subtract(arr1, arr2, dtype=np.float)

        t1 = torch.tensor(arr1)
        t2 = torch.tensor(arr2)
        t_res = torch.sub(t1, t2)
        np.testing.assert_allclose(np_res, t_res, rtol=1e-4, atol=0)

        v1 = self.cls(arr1)
        v2 = self.cls(arr2)
        v1 -= v2
        v_res = v1.result()

        np.testing.assert_allclose(np_res, v_res, rtol=1e-4, atol=0)

    def test_mul(self):
        arr1 = [random.random() for _ in range(100)]
        arr2 = [random.random() for _ in range(100)]

        np1 = np.array(arr1, dtype=np.float)
        np2 = np.array(arr2, dtype=np.float)
        np_res = np.multiply(arr1, arr2, dtype=np.float)

        t1 = torch.tensor(arr1)
        t2 = torch.tensor(arr2)
        t_res = torch.mul(t1, t2)
        np.testing.assert_allclose(np_res, t_res, rtol=1e-5, atol=0)

        v1 = self.cls(arr1)
        v2 = self.cls(arr2)
        v_res = (v1 * v2).result()

        np.testing.assert_allclose(np_res, v_res, rtol=1e-5, atol=0)

    def test_imul(self):
        arr1 = [random.random() for _ in range(100)]
        arr2 = [random.random() for _ in range(100)]

        np1 = np.array(arr1, dtype=np.float)
        np2 = np.array(arr2, dtype=np.float)
        np_res = np.multiply(arr1, arr2, dtype=np.float)

        t1 = torch.tensor(arr1)
        t2 = torch.tensor(arr2)
        t_res = torch.mul(t1, t2)
        np.testing.assert_allclose(np_res, t_res, rtol=1e-5, atol=0)

        v1 = self.cls(arr1)
        v2 = self.cls(arr2)
        v1 *= v2
        v_res = v1.result()

        np.testing.assert_allclose(np_res, v_res, rtol=1e-5, atol=0)

    def test_div(self):
        arr1 = [random.random() for _ in range(100)]
        arr2 = [random.random() for _ in range(100)]

        np1 = np.array(arr1, dtype=np.float)
        np2 = np.array(arr2, dtype=np.float)
        np_res = np.divide(arr1, arr2, dtype=np.float)

        t1 = torch.tensor(arr1)
        t2 = torch.tensor(arr2)
        t_res = torch.div(t1, t2)
        np.testing.assert_allclose(np_res, t_res, rtol=1e-5, atol=0)

        v1 = self.cls(arr1)
        v2 = self.cls(arr2)
        v_res = (v1 / v2).result()

        np.testing.assert_allclose(np_res, v_res, rtol=1e-5, atol=0)

    def test_idiv(self):
        arr1 = [random.random() for _ in range(100)]
        arr2 = [random.random() for _ in range(100)]

        np1 = np.array(arr1, dtype=np.float)
        np2 = np.array(arr2, dtype=np.float)
        np_res = np.divide(arr1, arr2, dtype=np.float)

        t1 = torch.tensor(arr1)
        t2 = torch.tensor(arr2)
        t_res = torch.div(t1, t2)
        np.testing.assert_allclose(np_res, t_res, rtol=1e-5, atol=0)

        v1 = self.cls(arr1)
        v2 = self.cls(arr2)
        v1 /= v2
        v_res = v1.result()

        np.testing.assert_allclose(np_res, v_res, rtol=1e-5, atol=0)


class TestEMVectorCuda(TestEMVector):
    cls = EMVectorCuda


class TestEMVectorOCL(TestEMVector):
    cls = EMVectorOCL


class TestEMVectorThreaded(TestEMVector):
    cls = EMVectorThreaded
