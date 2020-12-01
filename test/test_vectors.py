import unittest
import pytest
import numpy as np

from extreme_maths.vectors import EMVector
from extreme_maths.vectors import EMVectorCuda
from extreme_maths.vectors import EMVectorOCL
from extreme_maths.vectors import EMVectorThreaded


class TestEMVector(unittest.TestCase):
    cls = EMVector

    def test_add(self):
        rng = np.random.default_rng()
        arr1 = rng.random((100, ), dtype=np.float)
        arr2 = rng.random((100, ), dtype=np.float)
        arr_res = np.add(arr1, arr2, dtype=np.float)

        v1 = self.cls(arr1)
        v2 = self.cls(arr2)
        v_res = (v1 + v2).result()

        np.testing.assert_allclose(arr_res, v_res, rtol=1e-7, atol=0)

    def test_iadd(self):
        rng = np.random.default_rng()
        arr1 = rng.random((100, ), dtype=np.float)
        arr2 = rng.random((100, ), dtype=np.float)
        arr_res = np.add(arr1, arr2, dtype=np.float)

        v1 = self.cls(arr1)
        v2 = self.cls(arr2)
        v1 += v2
        v_res = v1.result()

        np.testing.assert_allclose(arr_res, v_res, rtol=1e-7, atol=0)

    def test_sub(self):
        rng = np.random.default_rng()
        arr1 = rng.random((100, ), dtype=np.float)
        arr2 = rng.random((100, ), dtype=np.float)
        arr_res = np.subtract(arr1, arr2, dtype=np.float)

        v1 = self.cls(arr1)
        v2 = self.cls(arr2)
        v_res = (v1 - v2).result()

        np.testing.assert_allclose(arr_res, v_res, rtol=1e-7, atol=0)

    def test_isub(self):
        rng = np.random.default_rng()
        arr1 = rng.random((100, ), dtype=np.float)
        arr2 = rng.random((100, ), dtype=np.float)
        arr_res = np.subtract(arr1, arr2, dtype=np.float)

        v1 = self.cls(arr1)
        v2 = self.cls(arr2)
        v1 -= v2
        v_res = v1.result()

        np.testing.assert_allclose(arr_res, v_res, rtol=1e-7, atol=0)

    def test_mul(self):
        rng = np.random.default_rng()
        arr1 = rng.random((100, ), dtype=np.float)
        arr2 = rng.random((100, ), dtype=np.float)
        arr_res = np.multiply(arr1, arr2, dtype=np.float)

        v1 = self.cls(arr1)
        v2 = self.cls(arr2)
        v_res = (v1 * v2).result()

        np.testing.assert_allclose(arr_res, v_res, rtol=1e-7, atol=0)

    def test_imul(self):
        rng = np.random.default_rng()
        arr1 = rng.random((100, ), dtype=np.float)
        arr2 = rng.random((100, ), dtype=np.float)
        arr_res = np.multiply(arr1, arr2, dtype=np.float)

        v1 = self.cls(arr1)
        v2 = self.cls(arr2)
        v1 *= v2
        v_res = v1.result()

        np.testing.assert_allclose(arr_res, v_res, rtol=1e-7, atol=0)

    def test_div(self):
        rng = np.random.default_rng()
        arr1 = rng.random((100, ), dtype=np.float)
        arr2 = rng.random((100, ), dtype=np.float)
        arr_res = np.divide(arr1, arr2, dtype=np.float)

        v1 = self.cls(arr1)
        v2 = self.cls(arr2)
        v_res = (v1 / v2).result()

        np.testing.assert_allclose(arr_res, v_res, rtol=1e-7, atol=0)

    def test_idiv(self):
        rng = np.random.default_rng()
        arr1 = rng.random((100, ), dtype=np.float)
        arr2 = rng.random((100, ), dtype=np.float)
        arr_res = np.divide(arr1, arr2, dtype=np.float)

        v1 = self.cls(arr1)
        v2 = self.cls(arr2)
        v1 /= v2
        v_res = v1.result()

        np.testing.assert_allclose(arr_res, v_res, rtol=1e-7, atol=0)


# class TestEMVectorCuda(TestEMVector):
# cls = EMVectorCuda

# class TestEMVectorOCL(TestEMVector):
# cls = EMVectorOCL

# class TestEMVectorThreaded(TestEMVector):
# cls = EMVectorThreaded
