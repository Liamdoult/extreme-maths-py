""" Vector maths """
import ctypes

import os

libdir = os.path.join(os.path.dirname(__file__), 'libem.so')

extreme_maths = ctypes.cdll.LoadLibrary(libdir)

_generate_vector = extreme_maths.generate_vector
_generate_vector.argtypes = [ctypes.c_int]


class _VECTOR(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int),
        ("vector", ctypes.c_int * 10),
    ]


_generate_vector.restype = _VECTOR


def generate_vector():
    return _generate_vector(ctypes.c_int(10))
