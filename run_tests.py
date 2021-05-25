"""
Run both unit and integration tests.
"""


from unittest import main as unittest_main

from tests.test_layers import *  # noqa: F401,F403,E501 pylint: disable=wildcard-import,unused-import,unused-wildcard-import
from tests.test_model import *  # noqa: F401,F403,E501 pylint: disable=wildcard-import,unused-import,unused-wildcard-import


if __name__ == '__main__':

    # running tests:
    unittest_main(verbosity=2)
