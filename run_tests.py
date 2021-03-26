"""
Run both unit and integration tests.
"""


from unittest import main as unittest_main

from tests.test_layers import TestEmbedder, TestLayerNorm, TestLogSoftmax,\
    TestMultiHeadAttention, TestPositionalEncoding,\
        TestPositionWiseFeedForward, TestResidualConnectionAndLayerNorm
from tests.test_model import TestModel


if __name__ == '__main__':
    raise Exception('Update all the imported tests!!')

    # running tests:
    unittest_main(verbosity=2)
