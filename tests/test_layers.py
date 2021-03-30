"""
Utilities for testing each single layer kind independently.
"""


from copy import deepcopy
from unittest import TestCase

from torch import float as torch_float, is_tensor, long as torch_long,\
    rand as torch_rand, randint as torch_randint
from torch.nn import Sequential

from transformer.architecture.attention import allowed_positions_to_attend,\
    MultiHeadAttention, scaled_dot_product_attention
from transformer.architecture.base import get_clones, LayerNorm, LogSoftmax,\
    PositionWiseFeedForward, ResidualConnectionAndLayerNorm
from transformer.architecture.embedding import Embedder, PositionalEncoding
from transformer.architecture.encoder import Encoder, EncoderBlock,\
    EncoderBlockBuildingBlocks
from transformer.architecture.decoder import Decoder, DecoderBlock,\
    DecoderBlockBuildingBlocks
from transformer.architecture.seq2seq import Seq2Seq, Seq2SeqBuildingBlocks
from transformer.training_and_inference.reproducibility import\
    make_results_reproducible


# hyperparameters for tests - all different enough to avoid coincidences:
SRC_VOCABULARY_DIMENSION = 90
TGT_VOCABULARY_DIMENSION = 110
N_ENCODER_BLOCKS = 5
N_DECODER_BLOCKS = 7
REPRESENTATION_DIMENSION = 64
FEEDFORWARD_DIMENSION = 256
N_ATTENTION_HEADS = 8
MAX_SEQUENCE_LENGTH = 30
DROPOUT_PROB = 0.1
MINI_BATCH_SIZE = 40


class ReproducibleTestLayer:
    """
    Common setups for reproducible tests.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        """
        Setup executed before every method (test) for reproducible results.
        """
        make_results_reproducible()


class StandardTestLayer:
    """
    Common standard shape and data type input-output tests.
    """

    def test_output_shapes_and_dtypes(self):
        """
        Test shape(s) and data type(s) of output tensor(s) given shape(s) and
        data types(s) of input tensor(s).
        """
        # asserting the test is correctly defined:
        assert len(self.expected_output_shapes) == len(
            self.expected_output_dtypes), "Ill-defined test."

        # computing output tensors:
        actual_output_tensors = self.layer(**self.forward_propagation_kwargs)

        # when only one single output tensor is returned:
        if is_tensor(actual_output_tensors):
            actual_output_tensors = [actual_output_tensors]

        # ensuring the expected number of outputs is returned:
        self.assertEqual(len(self.expected_output_shapes),
                         len(actual_output_tensors))

        # checking the shape and data type of each output tensor:
        for i, actual_output in enumerate(actual_output_tensors):

            # shape:

            subtest_name = "shape - output {n}".format(n=i)
            # extracting the actual and expected shapes of the output tensor:
            actual_shape = actual_output.shape
            expected_shape = self.expected_output_shapes[i]
            # checking the shape:
            with self.subTest(subtest_name):
                self.assertEqual(actual_shape, expected_shape)

            # data type:

            subtest_name = "dtype - output {n}".format(n=i)
            # extracting the actual and expected data types of the output
            # tensor:
            actual_dtype = actual_output.dtype
            expected_dtype = self.expected_output_dtypes[i]
            # checking the data type:
            with self.subTest(subtest_name):
                self.assertEqual(actual_dtype, expected_dtype)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Actual Tests Follow
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


class TestDecoder(ReproducibleTestLayer, StandardTestLayer, TestCase):
    """
    Tests for Decoder.
    """

    @classmethod
    def setUpClass(cls):
        """
        Avoid redundant, time-consuming, equivalent setups when testing across
        the different methods, that can use common instantiations.
        """
        feedforward_layer = PositionWiseFeedForward(
            token_representation_dimension=REPRESENTATION_DIMENSION,
            feedforward_dimension=FEEDFORWARD_DIMENSION,
            dropout_prob=DROPOUT_PROB
        )
        multi_head_attention_later = MultiHeadAttention(
            n_attention_heads=N_ATTENTION_HEADS,
            token_representation_dimension=REPRESENTATION_DIMENSION,
            dropout_prob=DROPOUT_PROB
        )
        cls.layer = Decoder(
            base_block=DecoderBlock(
                building_blocks=DecoderBlockBuildingBlocks(
                    self_multi_head_attention_layer=deepcopy(
                        multi_head_attention_later),
                    source_multi_head_attention_layer=deepcopy(
                        multi_head_attention_later),
                    fully_connected_layer=feedforward_layer
                ),
                feature_dimension=REPRESENTATION_DIMENSION,
                dropout_prob=DROPOUT_PROB
            ),
            n_clones=N_DECODER_BLOCKS
        )
        cls.forward_propagation_kwargs = {
            'tgt_features': torch_rand(
                size=(MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH - 1,
                      REPRESENTATION_DIMENSION),
                dtype=torch_float
            ),
            'src_encoded_tokens': torch_rand(
                size=(MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH,
                      REPRESENTATION_DIMENSION),
                dtype=torch_float
            ),
            'tgt_mask': torch_rand(
                size=(MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH - 1,
                      MAX_SEQUENCE_LENGTH - 1),
                dtype=torch_float
            ),
            'src_mask': torch_rand(
                size=(MINI_BATCH_SIZE, 1, MAX_SEQUENCE_LENGTH),
                dtype=torch_float
            )
        }
        cls.expected_output_shapes = [
            (MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH - 1,
             REPRESENTATION_DIMENSION)
        ]
        cls.expected_output_dtypes = [
            torch_float
        ]


class TestDecoderBlock(ReproducibleTestLayer, StandardTestLayer, TestCase):
    """
    Tests for DecoderBlock.
    """

    @classmethod
    def setUpClass(cls):
        """
        Avoid redundant, time-consuming, equivalent setups when testing across
        the different methods, that can use common instantiations.
        """
        feedforward_layer = PositionWiseFeedForward(
            token_representation_dimension=REPRESENTATION_DIMENSION,
            feedforward_dimension=FEEDFORWARD_DIMENSION,
            dropout_prob=DROPOUT_PROB
        )
        multi_head_attention_later = MultiHeadAttention(
            n_attention_heads=N_ATTENTION_HEADS,
            token_representation_dimension=REPRESENTATION_DIMENSION,
            dropout_prob=DROPOUT_PROB
        )
        cls.layer = DecoderBlock(
            building_blocks=DecoderBlockBuildingBlocks(
                self_multi_head_attention_layer=deepcopy(
                    multi_head_attention_later),
                source_multi_head_attention_layer=deepcopy(
                    multi_head_attention_later),
                fully_connected_layer=feedforward_layer
            ),
            feature_dimension=REPRESENTATION_DIMENSION,
            dropout_prob=DROPOUT_PROB
        )
        cls.forward_propagation_kwargs = {
            'tgt_features': torch_rand(
                size=(MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH - 1,
                      REPRESENTATION_DIMENSION),
                dtype=torch_float
            ),
            'src_encoded_tokens': torch_rand(
                size=(MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH,
                      REPRESENTATION_DIMENSION),
                dtype=torch_float
            ),
            'tgt_mask': torch_rand(
                size=(MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH - 1,
                      MAX_SEQUENCE_LENGTH - 1),
                dtype=torch_float
            ),
            'src_mask': torch_rand(
                size=(MINI_BATCH_SIZE, 1, MAX_SEQUENCE_LENGTH),
                dtype=torch_float
            )
        }
        cls.expected_output_shapes = [
            (MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH - 1,
             REPRESENTATION_DIMENSION)
        ]
        cls.expected_output_dtypes = [
            torch_float
        ]


class TestEncoder(ReproducibleTestLayer, StandardTestLayer, TestCase):
    """
    Tests for Encoder.
    """

    @classmethod
    def setUpClass(cls):
        """
        Avoid redundant, time-consuming, equivalent setups when testing across
        the different methods, that can use common instantiations.
        """
        feedforward_layer = PositionWiseFeedForward(
            token_representation_dimension=REPRESENTATION_DIMENSION,
            feedforward_dimension=FEEDFORWARD_DIMENSION,
            dropout_prob=DROPOUT_PROB
        )
        multi_head_attention_later = MultiHeadAttention(
            n_attention_heads=N_ATTENTION_HEADS,
            token_representation_dimension=REPRESENTATION_DIMENSION,
            dropout_prob=DROPOUT_PROB
        )
        cls.layer = Encoder(
            base_block=EncoderBlock(
                building_blocks=EncoderBlockBuildingBlocks(
                    self_multi_head_attention_layer=deepcopy(
                        multi_head_attention_later),
                    fully_connected_layer=feedforward_layer
                ),
                feature_dimension=REPRESENTATION_DIMENSION,
                dropout_prob=DROPOUT_PROB
            ),
            n_clones=N_ENCODER_BLOCKS
        )
        cls.forward_propagation_kwargs = {
            'src_features': torch_rand(
                size=(MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH,
                      REPRESENTATION_DIMENSION),
                dtype=torch_float
            ),
            'src_mask': torch_rand(
                size=(MINI_BATCH_SIZE, 1, MAX_SEQUENCE_LENGTH),
                dtype=torch_float
            )
        }
        cls.expected_output_shapes = [
            (MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH, REPRESENTATION_DIMENSION)
        ]
        cls.expected_output_dtypes = [
            torch_float
        ]


class TestEncoderBlock(ReproducibleTestLayer, StandardTestLayer, TestCase):
    """
    Tests for EncoderBlock.
    """

    @classmethod
    def setUpClass(cls):
        """
        Avoid redundant, time-consuming, equivalent setups when testing across
        the different methods, that can use common instantiations.
        """
        feedforward_layer = PositionWiseFeedForward(
            token_representation_dimension=REPRESENTATION_DIMENSION,
            feedforward_dimension=FEEDFORWARD_DIMENSION,
            dropout_prob=DROPOUT_PROB
        )
        multi_head_attention_later = MultiHeadAttention(
            n_attention_heads=N_ATTENTION_HEADS,
            token_representation_dimension=REPRESENTATION_DIMENSION,
            dropout_prob=DROPOUT_PROB
        )
        cls.layer = EncoderBlock(
            building_blocks=EncoderBlockBuildingBlocks(
                self_multi_head_attention_layer=deepcopy(
                    multi_head_attention_later),
                fully_connected_layer=feedforward_layer
            ),
            feature_dimension=REPRESENTATION_DIMENSION,
            dropout_prob=DROPOUT_PROB
        )
        cls.forward_propagation_kwargs = {
            'src_features': torch_rand(
                size=(MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH,
                      REPRESENTATION_DIMENSION),
                dtype=torch_float
            ),
            'src_mask': torch_rand(
                size=(MINI_BATCH_SIZE, 1, MAX_SEQUENCE_LENGTH),
                dtype=torch_float
            )
        }
        cls.expected_output_shapes = [
            (MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH, REPRESENTATION_DIMENSION)
        ]
        cls.expected_output_dtypes = [
            torch_float
        ]


class TestEmbedder(ReproducibleTestLayer, StandardTestLayer, TestCase):
    """
    Tests for Embedder.
    """

    @classmethod
    def setUpClass(cls):
        """
        Avoid redundant, time-consuming, equivalent setups when testing across
        the different methods, that can use common instantiations.
        """
        cls.layer = Embedder(
            token_representation_dimension=REPRESENTATION_DIMENSION,
            vocabulary_dimension=SRC_VOCABULARY_DIMENSION
        )
        cls.forward_propagation_kwargs = {
            'token_ids': torch_randint(
                low=0,
                high=SRC_VOCABULARY_DIMENSION,
                size=(MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH),
                dtype=torch_long
            )
        }
        cls.expected_output_shapes = [
            (MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH, REPRESENTATION_DIMENSION)
        ]
        cls.expected_output_dtypes = [
            torch_float
        ]


class TestLayerNorm(ReproducibleTestLayer, StandardTestLayer,
                    TestCase):
    """
    Tests for LayerNorm.
    """

    @classmethod
    def setUpClass(cls):
        """
        Avoid redundant, time-consuming, equivalent setups when testing across
        the different methods, that can use common instantiations.
        """
        cls.layer = LayerNorm(
            feature_dimension=REPRESENTATION_DIMENSION
        )
        cls.forward_propagation_kwargs = {
            'features': torch_rand(
                size=(MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH,
                      REPRESENTATION_DIMENSION),
                dtype=torch_float
            )
        }
        cls.expected_output_shapes = [
            (MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH, REPRESENTATION_DIMENSION)
        ]
        cls.expected_output_dtypes = [
            torch_float
        ]


class TestLogSoftmax(ReproducibleTestLayer, StandardTestLayer,
                     TestCase):
    """
    Tests for LogSoftmax.
    """

    @classmethod
    def setUpClass(cls):
        """
        Avoid redundant, time-consuming, equivalent setups when testing across
        the different methods, that can use common instantiations.
        """
        cls.layer = LogSoftmax(
            token_representation_dimension=REPRESENTATION_DIMENSION,
            vocabulary_dimension=TGT_VOCABULARY_DIMENSION
        )
        cls.forward_propagation_kwargs = {
            'logits': torch_rand(
                size=(MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH,
                      REPRESENTATION_DIMENSION),
                dtype=torch_float
            )
        }
        cls.expected_output_shapes = [
            (MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH, TGT_VOCABULARY_DIMENSION)
        ]
        cls.expected_output_dtypes = [
            torch_float
        ]


class TestMultiHeadAttention(ReproducibleTestLayer, StandardTestLayer,
                             TestCase):
    """
    Tests for MultiHeadAttention.
    """

    @classmethod
    def setUpClass(cls):
        """
        Avoid redundant, time-consuming, equivalent setups when testing across
        the different methods, that can use common instantiations.
        """
        cls.layer = MultiHeadAttention(
            n_attention_heads=N_ATTENTION_HEADS,
            token_representation_dimension=REPRESENTATION_DIMENSION,
            dropout_prob=DROPOUT_PROB
        )
        cls.forward_propagation_kwargs = {
            'query_tokens': torch_rand(
                size=(MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH - 1,
                      REPRESENTATION_DIMENSION),
                dtype=torch_float
            ),
            'key_or_value_tokens': torch_rand(
                size=(MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH,
                      REPRESENTATION_DIMENSION),
                dtype=torch_float
            ),
            'mask': torch_rand(
                size=(MINI_BATCH_SIZE, 1, MAX_SEQUENCE_LENGTH),
                dtype=torch_float
            )
        }
        cls.expected_output_shapes = [
            (MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH - 1,
                REPRESENTATION_DIMENSION)
        ]
        cls.expected_output_dtypes = [
            torch_float
        ]


class TestPositionalEncoding(ReproducibleTestLayer, StandardTestLayer,
                             TestCase):
    """
    Tests for PositionalEncoding.
    """

    @classmethod
    def setUpClass(cls):
        """
        Avoid redundant, time-consuming, equivalent setups when testing across
        the different methods, that can use common instantiations.
        """
        cls.layer = PositionalEncoding(
            token_representation_dimension=REPRESENTATION_DIMENSION,
            dropout_prob=DROPOUT_PROB,
            max_sequence_length=MAX_SEQUENCE_LENGTH
        )
        cls.forward_propagation_kwargs = {
            'token_embeddings': torch_rand(
                size=(MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH,
                      REPRESENTATION_DIMENSION),
                dtype=torch_float
            )
        }
        cls.expected_output_shapes = [
            (MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH, REPRESENTATION_DIMENSION)
        ]
        cls.expected_output_dtypes = [
            torch_float
        ]


class TestPositionWiseFeedForward(ReproducibleTestLayer, StandardTestLayer,
                                  TestCase):
    """
    Tests for PositionWiseFeedForward.
    """

    @classmethod
    def setUpClass(cls):
        """
        Avoid redundant, time-consuming, equivalent setups when testing across
        the different methods, that can use common instantiations.
        """
        cls.layer = PositionWiseFeedForward(
            token_representation_dimension=REPRESENTATION_DIMENSION,
            feedforward_dimension=FEEDFORWARD_DIMENSION,
            dropout_prob=DROPOUT_PROB
        )
        cls.forward_propagation_kwargs = {
            'features': torch_rand(
                size=(MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH,
                      REPRESENTATION_DIMENSION),
                dtype=torch_float
            )
        }
        cls.expected_output_shapes = [
            (MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH, REPRESENTATION_DIMENSION)
        ]
        cls.expected_output_dtypes = [
            torch_float
        ]


class TestResidualConnectionAndLayerNorm(ReproducibleTestLayer, StandardTestLayer,
                                         TestCase):
    """
    Tests for ResidualConnectionAndLayerNorm.
    """

    @classmethod
    def setUpClass(cls):
        """
        Avoid redundant, time-consuming, equivalent setups when testing across
        the different methods, that can use common instantiations.
        """
        cls.layer = ResidualConnectionAndLayerNorm(
            feature_dimension=REPRESENTATION_DIMENSION,
            dropout_prob=DROPOUT_PROB
        )
        position_wise_feed_forward = PositionWiseFeedForward(
            token_representation_dimension=REPRESENTATION_DIMENSION,
            feedforward_dimension=FEEDFORWARD_DIMENSION,
            dropout_prob=DROPOUT_PROB
        )
        cls.forward_propagation_kwargs = {
            'features': torch_rand(
                size=(MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH,
                      REPRESENTATION_DIMENSION),
                dtype=torch_float
            ),
            'base_layer_call': lambda x: position_wise_feed_forward(
                x
            )
        }
        cls.expected_output_shapes = [
            (MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH, REPRESENTATION_DIMENSION)
        ]
        cls.expected_output_dtypes = [
            torch_float
        ]


class TestSeq2Seq(ReproducibleTestLayer, StandardTestLayer, TestCase):
    """
    Tests for Seq2Seq.
    """

    @classmethod
    def setUpClass(cls):
        """
        Avoid redundant, time-consuming, equivalent setups when testing across
        the different methods, that can use common instantiations.
        """
        positional_encoding_layer = PositionalEncoding(
            token_representation_dimension=REPRESENTATION_DIMENSION,
            dropout_prob=DROPOUT_PROB,
            max_sequence_length=MAX_SEQUENCE_LENGTH
        )
        src_embedder = Sequential(
            Embedder(
                token_representation_dimension=REPRESENTATION_DIMENSION,
                vocabulary_dimension=SRC_VOCABULARY_DIMENSION
            ),
            deepcopy(positional_encoding_layer)
        )
        tgt_embedder = Sequential(
            Embedder(
                token_representation_dimension=REPRESENTATION_DIMENSION,
                vocabulary_dimension=TGT_VOCABULARY_DIMENSION
            ),
            deepcopy(positional_encoding_layer)
        )
        feedforward_layer = PositionWiseFeedForward(
            token_representation_dimension=REPRESENTATION_DIMENSION,
            feedforward_dimension=FEEDFORWARD_DIMENSION,
            dropout_prob=DROPOUT_PROB
        )
        multi_head_attention_later = MultiHeadAttention(
            n_attention_heads=N_ATTENTION_HEADS,
            token_representation_dimension=REPRESENTATION_DIMENSION,
            dropout_prob=DROPOUT_PROB
        )
        encoder = Encoder(
            base_block=EncoderBlock(
                building_blocks=EncoderBlockBuildingBlocks(
                    self_multi_head_attention_layer=deepcopy(
                        multi_head_attention_later),
                    fully_connected_layer=feedforward_layer
                ),
                feature_dimension=REPRESENTATION_DIMENSION,
                dropout_prob=DROPOUT_PROB
            ),
            n_clones=N_ENCODER_BLOCKS
        )
        decoder = Decoder(
            base_block=DecoderBlock(
                building_blocks=DecoderBlockBuildingBlocks(
                    self_multi_head_attention_layer=deepcopy(
                        multi_head_attention_later),
                    source_multi_head_attention_layer=deepcopy(
                        multi_head_attention_later),
                    fully_connected_layer=feedforward_layer
                ),
                feature_dimension=REPRESENTATION_DIMENSION,
                dropout_prob=DROPOUT_PROB
            ),
            n_clones=N_DECODER_BLOCKS
        )
        log_softmax_layer = LogSoftmax(
            token_representation_dimension=REPRESENTATION_DIMENSION,
            vocabulary_dimension=TGT_VOCABULARY_DIMENSION
        )
        building_blocks = Seq2SeqBuildingBlocks(
            encoder=encoder,
            decoder=decoder,
            src_embedder=src_embedder,
            tgt_embedder=tgt_embedder,
            log_softmax_layer=log_softmax_layer
        )
        cls.layer = Seq2Seq(
            building_blocks=building_blocks
        )
        cls.forward_propagation_kwargs = {
            'src_tokens': torch_randint(
                low=0,
                high=SRC_VOCABULARY_DIMENSION,
                size=(MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH),
                dtype=torch_long
            ),
            'tgt_tokens': torch_randint(
                low=0,
                high=TGT_VOCABULARY_DIMENSION,
                size=(MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH - 1),
                dtype=torch_long
            ),
            'src_mask': torch_rand(
                size=(MINI_BATCH_SIZE, 1, MAX_SEQUENCE_LENGTH),
                dtype=torch_float
            ),
            'tgt_mask': torch_rand(
                size=(MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH - 1,
                      MAX_SEQUENCE_LENGTH - 1),
                dtype=torch_float
            )
        }
        cls.expected_output_shapes = [
            (MINI_BATCH_SIZE, MAX_SEQUENCE_LENGTH - 1,
             REPRESENTATION_DIMENSION)
        ]
        cls.expected_output_dtypes = [
            torch_float
        ]
