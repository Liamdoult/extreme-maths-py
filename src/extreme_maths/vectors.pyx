cimport cython
from libc.stdlib cimport malloc, free

cimport extreme_maths.cem as cem

cdef class Vector:
    cdef float* _array
    cdef int _len

    def __len__(self):
        return self._len

    def add(self, other: Vector):
        cdef float *result = cem.add_f(self._array, other._array, <int *>&self._len)

        vec = Vector()
        vec._array = result
        vec._len = self._len
        return vec

    def __add__(self, other):
        return self.add(other)

    def iadd(self, other: Vector):
        result = cem.iadd_f(self._array, other._array, <int *>&self._len)
        return self

    def __iadd__(self, other):
        return self.iadd(other)

    def sub(self, other: Vector):
        cdef float *result = cem.sub_f(self._array, other._array, <int *>&self._len)

        vec = Vector()
        vec._array = result
        vec._len = self._len
        return vec

    def __sub__(self, other):
        return self.sub(other)

    def isub(self, other: Vector):
        result = cem.isub_f(self._array, other._array, <int *>&self._len)
        return self

    def __isub__(self, other):
        return self.isub(other)

    def mul(self, other: Vector):
        cdef float *result = cem.mul_f(self._array, other._array, <int *>&self._len)

        vec = Vector()
        vec._array = result
        vec._len = self._len
        return vec

    def __mul__(self, other):
        return self.mul(other)

    def imul(self, other: Vector):
        result = cem.imul_f(self._array, other._array, <int *>&self._len)
        return self

    def __imul__(self, other):
        return self.imul(other)

    def div(self, other: Vector):
        cdef float *result = cem.div_f(self._array, other._array, <int *>&self._len)

        vec = Vector()
        vec._array = result
        vec._len = self._len
        return vec

    def __truediv__(self, other):
        return self.div(other)

    def idiv(self, other: Vector):
        result = cem.idiv_f(self._array, other._array, <int *>&self._len)
        return self

    def __itruediv__(self, other):
        return self.idiv(other)

    def result(self):
        return [self._array[i] for i in range(len(self))]

    def __dealloc__(self):
        free(self._array)

def vector(array):
    vec = Vector()
    vec._array = <float *>malloc(<int>len(array)*cython.sizeof(float))
    vec._len = <int>len(array)

    for i in xrange(len(array)):
        vec._array[i] = array[i]

    return vec
