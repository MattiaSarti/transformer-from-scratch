"""
Run both unit and integration tests.
"""


from unittest import main as unittest_main

from tests.test_layers import *  # noqa: F401,F403,E501 pylint: disable=W0401,W0611,W0614
from tests.test_model import *  # noqa: F401,F403,E501 pylint: disable=W0401,W0611,W0614


if __name__ == '__main__':

    # running tests:
    unittest_main(verbosity=2)
