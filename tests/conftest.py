"""Shared pytest setup: make ``src/`` importable without installing the package.

The insert must happen at import time of this file (collection), because the test
modules import ``mmprog`` at module level, before any fixture runs.
"""
import os
import sys

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@pytest.fixture(autouse=True, scope="session")
def repo_src_on_path():
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    yield
