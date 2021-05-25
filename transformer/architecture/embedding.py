"""
Embedding and positional encoding layers.
"""


from math import log, sqrt

from torch import arange as torch_arange, cos as torch_cos, exp as torch_exp,\
    sin as torch_sin, Tensor, zeros as torch_zeros  # noqa: E501 pylint: disable=E0611
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

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Forward propagation.

        Tensor Shapes:

            Args:
                token_ids: (batch size, sequence length)

            Returns:
                (batch size, sequence length, n. features)

        """
        return self.core_embedding_layer(token_ids) * \
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

    def forward(self, token_embeddings) -> Tensor:
        """
        Forward propagation.

        Tensor Shapes:

            Args:
                token_embeddings: (batch size, sequence length, n. features)

            Returns:
                (batch size, sequence length, n. features)

        """
        return self.dropout_layer(
            token_embeddings +
            self.positional_signals[:, :token_embeddings.size(1)]
            # positional signal added only over sequence
        )
