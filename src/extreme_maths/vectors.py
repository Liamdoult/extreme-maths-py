from array import array
import atexit
import ctypes
import functools
import os


class _Vector(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int),
        ("array", ctypes.c_void_p),
    ]


def load_custom_types(lib_name):
    lib_path = os.path.join(os.path.dirname(__file__), f"{lib_name}.so")
    lib = ctypes.cdll.LoadLibrary(lib_path)

    lib.init()

    # Ensure proper shutdown always
    atexit.register(lib.close)

    lib.create_f.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.create_f.restype = _Vector
    setattr(_Vector, f"create_f", staticmethod(lib.create_f))

    lib.result_f.argtypes = [ctypes.POINTER(_Vector)]
    lib.result_f.restype = ctypes.POINTER(ctypes.c_float)
    func = functools.partialmethod(lib.result_f)
    setattr(_Vector, f"result", func)

    lib.clean_f.argtypes = [ctypes.POINTER(_Vector)]
    lib.clean_f.restype = None
    func = functools.partialmethod(lib.clean_f)
    setattr(_Vector, f"clean", func)

    for op in ["add", "iadd", "sub", "isub", "mul", "imul", "div", "idiv"]:
        for t in ["float"]:
            func = getattr(lib, f"{op}_f")
            func.argtypes = [
                ctypes.POINTER(_Vector),
                ctypes.POINTER(_Vector),
            ]
            if op[0] == "i":
                func.restype = None
            else:
                func.restype = _Vector

            func = functools.partialmethod(func)
            setattr(_Vector, f"{op}", func)


class EMVector:
    def __init__(self, arr):
        if isinstance(arr, _Vector):
            self._array = arr
        else:
            self._array = _Vector.create_f(
                (ctypes.c_float * len(arr)).from_buffer(array('f', arr)),
                ctypes.c_int(len(arr)))

    def __len__(self):
        return self._array.size

    def add(self, other):
        return self.__class__(self._array.add(other._array))
        raise TypeError(
            f"Operation not supported between {self.__class__} and {other.__class__}"
        )

    def __add__(self, other):
        return self.add(other)

    def iadd(self, other):
        self._array.iadd(other._array)
        return self
        raise TypeError(
            f"Operation not supported between {self.__class__} and {other.__class__}"
        )

    def __iadd__(self, other):
        return self.iadd(other)

    def sub(self, other):
        return self.__class__(self._array.sub(other._array))
        raise TypeError(
            f"Operation not supported between {self.__class__} and {other.__class__}"
        )

    def __sub__(self, other):
        return self.sub(other)

    def isub(self, other):
        self._array.isub(other._array)
        return self
        raise TypeError(
            f"Operation not supported between {self.__class__} and {other.__class__}"
        )

    def __isub__(self, other):
        return self.isub(other)

    def mul(self, other):
        return self.__class__(self._array.mul(other._array))
        raise TypeError(
            f"Operation not supported between {self.__class__} and {other.__class__}"
        )

    def __mul__(self, other):
        return self.mul(other)

    def imul(self, other):
        self._array.imul(other._array)
        return self
        raise TypeError(
            f"Operation not supported between {self.__class__} and {other.__class__}"
        )

    def __imul__(self, other):
        return self.imul(other)

    def div(self, other):
        return self.__class__(self._array.div(other._array))
        raise TypeError(
            f"Operation not supported between {self.__class__} and {other.__class__}"
        )

    def __truediv__(self, other):
        return self.div(other)

    def idiv(self, other):
        self._array.idiv(other._array)
        return self
        raise TypeError(
            f"Operation not supported between {self.__class__} and {other.__class__}"
        )

    def __itruediv__(self, other):
        return self.idiv(other)

    def __del__(self):
        self._array.clean()

    def result(self):
        res = self._array.result()
        return [res[i] for i in range(self._array.size)]
        # for i in range(self._array.size):
        #     yield res[i]


load_custom_types("libem")
