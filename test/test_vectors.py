import pytest

from extreme_maths.vectors import generate_vector


def test_generate_vector():
    vector = generate_vector()
    print(vector.size)
    assert vector.size == 10
    for i in range(10):
        vector.vector[i] = i
    assert vector.vector == range(10)
