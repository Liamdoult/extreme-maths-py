import ctypes
import os


class _VECTOR(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int),
        ("array", ctypes.c_void_p),
    ]


libem = os.path.join(os.path.dirname(__file__), 'libem.so')
libem_cuda = os.path.join(os.path.dirname(__file__), 'libem_cuda.so')
libem_ocl = os.path.join(os.path.dirname(__file__), 'libem_ocl.so')
libem_threaded = os.path.join(os.path.dirname(__file__), 'libem_threaded.so')

em = ctypes.cdll.LoadLibrary(libem)
em_cuda = ctypes.cdll.LoadLibrary(libem_cuda)
em_ocl = ctypes.cdll.LoadLibrary(libem_ocl)
em_threaded = ctypes.cdll.LoadLibrary(libem_threaded)
for lib in [em, em_cuda, em_ocl, em_threaded]:
    lib.init()

    lib.create_vector.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.create_vector.restype = _VECTOR

    lib.get_result.argtypes = [ctypes.POINTER(_VECTOR)]
    lib.get_result.restype = ctypes.POINTER(ctypes.c_float)

    lib.clean_vector.argtypes = [_VECTOR]
    lib.clean_vector.restype = None

    lib.vector_add.argtypes = [
        ctypes.POINTER(_VECTOR),
        ctypes.POINTER(_VECTOR)
    ]
    lib.vector_add.restype = _VECTOR

    lib.vector_iadd.argtypes = [
        ctypes.POINTER(_VECTOR),
        ctypes.POINTER(_VECTOR)
    ]
    lib.vector_iadd.restype = None

    lib.vector_sub.argtypes = [
        ctypes.POINTER(_VECTOR),
        ctypes.POINTER(_VECTOR)
    ]
    lib.vector_sub.restype = _VECTOR

    lib.vector_isub.argtypes = [
        ctypes.POINTER(_VECTOR),
        ctypes.POINTER(_VECTOR)
    ]
    lib.vector_isub.restype = None

    lib.vector_mul.argtypes = [
        ctypes.POINTER(_VECTOR),
        ctypes.POINTER(_VECTOR)
    ]
    lib.vector_mul.restype = _VECTOR

    lib.vector_imul.argtypes = [
        ctypes.POINTER(_VECTOR),
        ctypes.POINTER(_VECTOR)
    ]
    lib.vector_imul.restype = None

    lib.vector_div.argtypes = [
        ctypes.POINTER(_VECTOR),
        ctypes.POINTER(_VECTOR)
    ]
    lib.vector_div.restype = _VECTOR

    lib.vector_idiv.argtypes = [
        ctypes.POINTER(_VECTOR),
        ctypes.POINTER(_VECTOR)
    ]
    lib.vector_idiv.restype = None


class _EMVector:
    def __init__(self, array):
        if not isinstance(array, _VECTOR):
            array = self._lib.create_vector(
                (ctypes.c_float * len(array))(*array), len(array))
        self._array = array

    def __len__(self):
        return self._array.size

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Addition between EMVector and `{type(other)}` not supported")
        return self.__class__(self._lib.vector_add(self._array, other._array))

    def __iadd__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Addition between EMVector and `{type(other)}` not supported")
        self._lib.vector_iadd(self._array, other._array)
        return self

    def __sub__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Addition between EMVector and `{type(other)}` not supported")
        return self.__class__(self._lib.vector_sub(self._array, other._array))

    def __isub__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Addition between EMVector and `{type(other)}` not supported")
        self._lib.vector_isub(self._array, other._array)
        return self

    def __mul__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Addition between EMVector and `{type(other)}` not supported")
        return self.__class__(self._lib.vector_mul(self._array, other._array))

    def __imul__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Addition between EMVector and `{type(other)}` not supported")
        self._lib.vector_imul(self._array, other._array)
        return self

    def __truediv__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Addition between EMVector and `{type(other)}` not supported")
        return self.__class__(self._lib.vector_div(self._array, other._array))

    def __itruediv__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Addition between EMVector and `{type(other)}` not supported")
        self._lib.vector_idiv(self._array, other._array)
        return self

    def __del__(self):
        self._lib.clean_vector(self._array)

    def result(self):
        res = self._lib.get_result(self._array)
        return [res[i] for i in range(self._array.size)]


class EMVector(_EMVector):
    _lib = em


class EMVectorCuda(_EMVector):
    _lib = em_cuda


class EMVectorOCL(_EMVector):
    _lib = em_ocl


class EMVectorThreaded(EMVector):
    _lib = em_threaded
