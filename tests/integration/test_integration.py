#!python3
# -*- coding: utf-8 -*-
"""
This is an integration test example
Integration tests verify the interaction between multiple components or external systems (e.g., databases, APIs).
"""

from mobts import hello_world


def test_hello_world():
    assert hello_world() is None
