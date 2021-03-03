"""
Embedding and positional encoding layers.
"""


from math import log, sqrt

from torch import arange as torch_arange
from torch import cos as torch_cos
from torch import exp as torch_exp
from torch import Tensor
from torch import sin as torch_sin
from torch import zeros as torch_zeros
from torch.nn import Dropout, Embedding, Module


class Embedder(Module):
    """
    Embedding layer that, besides pure embedding, additionally carries out the
    (element-wise) multiplication of the embedded feature vector by the square
    root of the embedding dimension size.
    """
    def __init__(self, vocabulary_dimension: int,
                 token_representation_dimension: int) -> None:
        super(Embedder, self).__init__()
        self.core_embedding_layer = Embedding(
            num_embeddings=vocabulary_dimension,
            embedding_dim=token_representation_dimension
        )
        self.token_representation_dimension = token_representation_dimension

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward propagation.
        """
        return self.core_embedding_layer(x) * \
            sqrt(self.token_representation_dimension)


class PositionalEncoding(Module):
    """
    Positional encoding layer, adding position information to feature values
    of input embeddings and eventually applying dropout.
    """
    def __init__(self, token_representation_dimension: int, dropout_prob:
                 float, max_sequence_length: int) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout_layer = Dropout(p=dropout_prob)

        # defining positional signals added to embeddings:

        # initialization:
        positional_signals = torch_zeros(
            (max_sequence_length, token_representation_dimension),
            requires_grad=False
        )

        positions = torch_arange(
            start=0,
            end=max_sequence_length,
            requires_grad=False
        ).unsqueeze(dim=1)

        wave_inputs = positions * torch_exp(
            torch_arange(
                start=0, end=token_representation_dimension, step=2
            ) * (-log(10000.0) / token_representation_dimension)
        )  # ✓ see demonstration on my notes ▢

        # interleaving sinusoidal and cosinusoidal components along feature
        # dimension (starting with sine), yielding positional signals for
        # all the allowed positions (for sequences up to the maximum allowed
        # length):
        positional_signals[:, 0::2] = torch_sin(wave_inputs)
        positional_signals[:, 1::2] = torch_cos(wave_inputs)
        positional_signals = positional_signals.unsqueeze(dim=0)

        # parameters not requiring backpropagation (i.e. gradient computation
        # and update):
        self.register_buffer('positional_signals', positional_signals)
        # TODO: understand if redundant with requires_grad=False

    def forward(self, x) -> Tensor:
        """
        Forward propagation.
        """
        return self.dropout_layer(
            x + self.positional_signals[:, :x.size(1)]  # only over sequence
        )
