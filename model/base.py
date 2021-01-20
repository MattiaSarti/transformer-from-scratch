from copy import deepcopy
from math import log, sqrt

from numpy import ones as np_ones
from numpy import tril
from torch import arange as torch_arange
from torch import cos as torch_cos
from torch import cos as torch_exp
from torch import from_numpy, matmul, nn, tensor, Tensor
from torch import ones as torch_ones
from torch import sin as torch_sin
from torch import zeros as torch_zeros
from torch.autograd import Variable
from torch.nn import functional as F


def get_clones(module_to_be_cloned, n_clones) -> nn.ModuleList:
    """
    Produce 'n' "architecturally" identical layers, without shared parameters.
    """
    return nn.ModuleList(
        [deepcopy(module_to_be_cloned) for _ in range(n_clones)]
    )
    
    
class LogSoftmax(nn.Module):
    """
    Linear layer followed by log-softmax activation function.
    """
    def __init__(self, token_representation_dimension: int, vocabulary_dimension: int):
        super(LogSoftmax, self).__init__()
        self.linear_layer = nn.Linear(in_features=token_representation_dimension,
            out_features=vocabulary_dimension)

    def forward(self, x: Tensor) -> Tensor:
        return F.log_softmax(self.linear_layer(x), dim=-1)


class LayerNorm(nn.Module):
    """
    Layer-normalization layer.
    """
    def __init__(self, feature_dimension: int, epsilon: float=1e-6):
        super(LayerNorm, self).__init__()
        self.alpha = nn.Parameter(data=torch_ones(feature_dimension))
        self.beta = nn.Parameter(data=torch_zeros(feature_dimension))
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
       mean = x.mean(dim=-1, keepdim=True)
       std = x.std(dim=-1, keepdim=True)
       return ((x - mean) / (std + self.epsilon)) * self.alpha + self.beta


class ResidualConnectionAndLayerNorm(nn.Module):
    """
    Residual connection around a base layer, after dropout is applied to the 
    base layer's output, eventually followed by layer-normalization.
    """
    def __init__(self, feature_dimension: int, dropout_prob: float):
        super(ResidualConnectionAndLayerNorm, self).__init__()
        self.layer_normalization_layer = LayerNorm(feature_dimension=\
            feature_dimension)
        self.dropout_layer = nn.Dropout(p=dropout_prob)
        
    def forward(self, x, base_layer: nn.Module) -> Tensor:
        return self.layer_normalization_layer(x + self.dropout_layer(base_layer(x)))
    # TODO: understand why norm is applied first instead of last in their implementation: "Note for code simplicity the norm is first as opposed to last."


class PositionWiseFeedForward(nn.Module):
    """
    Point-wise feed-forward (fully-connected) layer:
    - parameters (weights) are shared among different positions of the same 
        layer, but not among different layers;
    - equation: FFN(x) = max(0, xW1 + b1)W2 + b2 , with dropout applied only right
        after ReLU (i.e. max(..., 0)) application, during training.
    """
    def __init__(self,
                 token_representation_dimension: int,
                 feedforward_dimension: int,
                 dropout_prob: float):
        super(PositionWiseFeedForward, self).__init__()
        self.linear_layer_1 = nn.Linear(
            in_features=token_representation_dimension,
            out_features=feedforward_dimension
        )
        self.linear_layer_2 = nn.Linear(
            in_features=feedforward_dimension,
            out_features=token_representation_dimension
        )
        self.dropout_layer = nn.Dropout(dropout_prob)

        def forward(self, x: Tensor) -> Tensor:
            return self.linear_layer_2(
                self.dropout_layer(F.relu(self.linear_layer_1(x)))
            )
