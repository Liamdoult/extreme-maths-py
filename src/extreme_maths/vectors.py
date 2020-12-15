from array import array
import atexit
import ctypes
import functools
import os


def load_custom_types(lib_name):
    lib_path = os.path.join(os.path.dirname(__file__), f"{lib_name}.so")
    lib = ctypes.cdll.LoadLibrary(lib_path)

    lib.init()

    # Ensure proper shutdown always
    atexit.register(lib.close)

    lib.copy_f.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.copy_f.restype = _Vector
    setattr(_Vector, f"copy_f", staticmethod(lib.copy_f))

    lib.point_f.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.point_f.restype = _Vector
    setattr(_Vector, f"point_f", staticmethod(lib.point_f))

    lib.result_f.argtypes = [ctypes.POINTER(_Vector)]
    lib.result_f.restype = ctypes.POINTER(ctypes.c_float)
    func = functools.partialmethod(lib.result_f)
    setattr(_Vector, f"_result", func)

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

                def exec(self, op, other):
                    op(self, other)
                    return self

                func = functools.partialmethod(exec, func)
            else:
                func.restype = _Vector

                def exec(self, op, other):
                    return op(self, other)

                func = functools.partialmethod(exec, func)

            setattr(_Vector, f"{op}", func)


class _Vector(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int),
        #("array", ctypes.c_void_p),
        ("array", ctypes.POINTER(ctypes.c_float)),
    ]

    def __add__(self, other):
        return self.add(other)

    def __iadd__(self, other):
        return self.iadd(other)

    def __sub__(self, other):
        return self.sub(other)

    def __isub__(self, other):
        return self.isub(other)

    def __mul__(self, other):
        return self.mul(other)

    def __imul__(self, other):
        return self.imul(other)

    def __truediv__(self, other):
        return self.div(other)

    def __itruediv__(self, other):
        return self.idiv(other)

    # def __del__(self):
    # self.clean()

    def result(self):
        res = self._result()
        return [res[i] for i in range(self.size)]


def vector(arr):
    return _Vector.copy_f(
        (ctypes.c_float * len(arr)).from_buffer_copy(array('f', arr)),
        ctypes.c_int(len(arr)))


load_custom_types("libem")
