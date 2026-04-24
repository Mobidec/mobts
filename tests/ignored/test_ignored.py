#!python3
# -*- coding: utf-8 -*-
"""
This test is ignored when launching pytest command because located in tests/ignored (as configured in tox.ini).
"""

from mobts import hello_world


def test_hello_world():
    assert hello_world() is None
