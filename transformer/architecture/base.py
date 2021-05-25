"""
Base layers and utilities.
"""


from copy import deepcopy
from typing import Callable

from torch import ones as torch_ones, Tensor, zeros as torch_zeros  # noqa: E501 pylint: disable=E0611
from torch.nn import Dropout, Linear, Module, ModuleList, Parameter
from torch.nn.functional import log_softmax, relu


def get_clones(module_to_be_cloned, n_clones) -> ModuleList:
    """
    Produce 'n' "architecturally" identical layers, without shared parameters.
    """
    return ModuleList(
        [deepcopy(module_to_be_cloned) for _ in range(n_clones)]
    )


class LogSoftmax(Module):
    """
    Linear layer followed by log-softmax activation function.
    """

    def __init__(self, token_representation_dimension: int,
                 vocabulary_dimension: int) -> None:
        super(LogSoftmax, self).__init__()
        self.linear_layer = Linear(
            in_features=token_representation_dimension,
            out_features=vocabulary_dimension
        )

    def forward(self, logits: Tensor) -> Tensor:
        """
        Forward propagation.

        Tensor Shapes:

            Args:
                logits: (batch size, sequence length - 1, n. features)

            Returns:
                (batch size, sequence length - 1, vocabulary size)

        """
        # NOTE: the linear layer transformation is applied to the last channel
        # of the n-dimensional (n > 2) tensor, i.e. separately for each token
        # position (and so does the log-softmax activation function as well):
        return log_softmax(self.linear_layer(logits), dim=-1)


class LayerNorm(Module):
    """
    Layer-normalization layer.
    """

    def __init__(self, feature_dimension: int, epsilon: float = 1e-6) -> None:
        super(LayerNorm, self).__init__()
        self.alpha = Parameter(data=torch_ones((feature_dimension)))
        self.beta = Parameter(data=torch_zeros((feature_dimension)))
        self.epsilon = epsilon

    def forward(self, features: Tensor) -> Tensor:
        """
        Forward propagation.

        Tensor Shapes:

            Args:
                features: (batch size, sequence length | sequence length - 1,
                    n. features)

            Returns:
                (batch size, sequence length | sequence length - 1,
                    n. features)

        """
        mean = features.mean(dim=-1, keepdim=True)
        std = features.std(dim=-1, keepdim=True)
        return ((features - mean) / (std + self.epsilon)) * self.alpha\
            + self.beta


class ResidualConnectionAndLayerNorm(Module):
    """
    Residual connection around a base layer, after dropout is applied to the
    base layer's output, eventually followed by layer-normalization.
    """

    def __init__(self, feature_dimension: int, dropout_prob: float) -> None:
        super(ResidualConnectionAndLayerNorm, self).__init__()
        self.layer_normalization_layer = LayerNorm(
            feature_dimension=feature_dimension
        )
        self.dropout_layer = Dropout(p=dropout_prob)

    def forward(self, features, base_layer_call: Callable[[Tensor, Tensor],
                                                          Tensor]) -> Tensor:
        """
        Forward propagation.

        Tensor Shapes:

            Args:
                features: (batch size, sequence length | sequence length - 1,
                    n. features)

            Returns:
                (batch size, sequence length | sequence length - 1,
                    n. features)

        """
        return features + self.dropout_layer(
            base_layer_call(
                self.layer_normalization_layer(features)
            )
        )
    # NOTE: normalization is applied first instead of last in their
    # implementation: "Note for code simplicity the norm is first as opposed
    # to last."


class PositionWiseFeedForward(Module):
    """
    Point-wise feed-forward (fully-connected) layer:
    - parameters (weights) are shared among different positions of the same
        layer, but not among different layers;
    - equation: FFN(x) = max(0, xW1 + b1)W2 + b2 , with dropout applied only
        right after ReLU (i.e. max(..., 0)) application, during training.
    """

    def __init__(self, token_representation_dimension: int,
                 feedforward_dimension: int, dropout_prob: float) -> None:
        super(PositionWiseFeedForward, self).__init__()
        self.linear_layer_1 = Linear(
            in_features=token_representation_dimension,
            out_features=feedforward_dimension
        )
        self.linear_layer_2 = Linear(
            in_features=feedforward_dimension,
            out_features=token_representation_dimension
        )
        self.dropout_layer = Dropout(dropout_prob)

    def forward(self, features: Tensor) -> Tensor:
        """
        Forward propagation.

        Tensor Shapes:

            Args:
                features: (batch size, sequence length | sequence length - 1,
                    n. features)

            Returns:
                (batch size, sequence length | sequence length - 1,
                    n. features)

        """
        return self.linear_layer_2(
            self.dropout_layer(relu(self.linear_layer_1(features)))
        )
