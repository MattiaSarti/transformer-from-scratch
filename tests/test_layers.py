"""
Utilities for testing each single layer kind independently.
"""


from unittest import TestCase

from torch import long as torch_long, rand as torch_rand, tensor

from transformer.architecture.attention import allowed_positions_to_attend,\
    MultiHeadAttention, scaled_dot_product_attention
from transformer.architecture.base import get_clones, LayerNorm, LogSoftmax,\
    PositionWiseFeedForward, ResidualConnectionAndLayerNorm
from transformer.architecture.embedding import Embedder, PositionalEncoding
from transformer.architecture.encoder import Encoder, EncoderBlock
from transformer.architecture.decoder import Decoder, DecoderBlock
from transformer.architecture.seq2seq import Seq2Seq


# hyperparameters for tests:
SRC_VOCABULARY_DIMENSION = 9000
TGT_VOCABULARY_DIMENSION = 11000
N_ENCODER_BLOCKS = 5
N_DECODER_BLOCKS = 7
REPRESENTATION_DIMENSION = 512
FEEDFORWARD_DIMENSION = 2048
N_ATTENTION_HEADS = 8
MAX_SEQUENCE_LENGTH = 300
DROPOUT_PROB = 0.1
MINI_BATCH_SIZE = 40


class TestEmbedder(TestCase):

    def setUpClass(self):
        """
        Avoid redundant, time-consuming, equivalent setups when testing across
        the different methods, that can use common instantiations.
        """
        self.layer = Embedder(
            token_representation_dimension=REPRESENTATION_DIMENSION,
            vocabulary_dimension=SRC_VOCABULARY_DIMENSION
        )
        self.input_tensor = torch_rand()
    
    # def setUp(self):
    #     """
    #     Setup executed before every method (test).
    #     """

    def test_tensor_shapes(self):
        """
        Test shapes of both input and output tensors.
        """
        with self.subTest('input'):
            self.assertTrue(False)

        with self.subTest('output'):
            self.assertTrue(False)

    def test_tensor_dtypes(self):
        """
        Test data types of both input and output tensors.
        """
        with self.subTest('input'):
            self.assertTrue(False)

        with self.subTest('output'):
            self.assertTrue(False)


class TestPositionalEncoding(TestCase):
    
    
        self.layer = PositionalEncoding(
            token_representation_dimension=REPRESENTATION_DIMENSION,
            dropout_prob=DROPOUT_PROB,
            max_sequence_length=MAX_SEQUENCE_LENGTH
        )

    def test_tensor_shapes(self):
        """
        Test shapes of both input and output tensors.
        """
        with self.subTest('input'):
            self.assertTrue(False)

        with self.subTest('output'):
            self.assertTrue(False)

    def test_tensor_dtypes(self):
        """
        Test data types of both input and output tensors.
        """
        with self.subTest('input'):
            self.assertTrue(False)

        with self.subTest('output'):
            self.assertTrue(False)
