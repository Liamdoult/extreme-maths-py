# Extreme Maths Python

![Python tests](https://github.com/Liamdoult/extreme-maths-py/workflows/Python%20tests/badge.svg)

The plan... simple. Be faster.

![Average speed compared to Numpy and Pytorch](https://github.com/Liamdoult/extreme-maths-py/blob/master/docs/average.png)

## Build

Build Cython libraries:

    python setup.py build_ext -i

## Testing

All tests are written in with pytest. You need to build `extreme_maths_c` and copy the `libem.so` binary into `src/extreme_maths/` to use or test the library.

    $ tox

To run the benchmark test (Not run by default):

    pytest -s -v test/test_perf.py
