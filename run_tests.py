"""
Run both unit and integration tests.
"""


from unittest import main as unittest_main

from tests.test_layers import *
from tests.test_model import *


if __name__ == '__main__':

    # running tests:
    unittest_main()
