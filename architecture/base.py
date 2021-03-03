"""
Base layers and utilities.
"""


from copy import deepcopy

from torch import Tensor
from torch import ones as torch_ones
from torch import zeros as torch_zeros
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward propagation definition.
        """
        return log_softmax(self.linear_layer(x), dim=-1)


class LayerNorm(Module):
    """
    Layer-normalization layer.
    """
    def __init__(self, feature_dimension: int, epsilon: float = 1e-6) -> None:
        super(LayerNorm, self).__init__()
        self.alpha = Parameter(data=torch_ones(feature_dimension))
        self.beta = Parameter(data=torch_zeros(feature_dimension))
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward propagation definition.
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return ((x - mean) / (std + self.epsilon)) * self.alpha + self.beta


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

    def forward(self, x, base_layer: Module) -> Tensor:
        """
        Forward propagation definition.
        """
        return x + self.dropout_layer(
            base_layer(
                self.layer_normalization_layer(x)
            )
        )
    # TODO: understand why norm is applied first instead of last in their
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward propagation definition.
        """
        return self.linear_layer_2(
            self.dropout_layer(relu(self.linear_layer_1(x)))
        )
