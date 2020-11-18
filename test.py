from ctypes import *

extreme_maths = cdll.LoadLibrary("./libem.so")

custom_add = extreme_maths.add
custom_add.argtypes = [POINTER(c_int), POINTER(c_int)]
custom_add.restype = c_int

print(custom_add(pointer(c_int(10)), pointer(c_int(10))))

_generate_vector = extreme_maths.generate_vector
_generate_vector.argtypes = [c_int]


# https://stackoverflow.com/questions/8392203/dynamic-arrays-and-structures-in-structures-in-python
class _VECTOR(Structure):
    _fields_ = [
        ("size", c_int),
        ("vector", c_int * 1),
    ]


def generate_vector(n: int):
    _generate_vector.restype = _VECTOR
    return generate_vector(c_int(n))


vector = generate_vector(10)
print([i for i in vector.vector])
vector = generate_vector(20)
print([i for i in vector.vector])
