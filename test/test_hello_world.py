import pytest

from extreme_maths import template_module


def test_main():
    from extreme_maths import __main__


def test_hello_world():
    """ This is an example of a basic test. """
    assert template_module.hello_world() is None
