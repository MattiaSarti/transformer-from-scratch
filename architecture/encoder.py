"""
Encoder architecture.
"""


from torch import Tensor
from torch.nn import Module

from transformer.architecture.base import get_clones, LayerNorm,\
    ResidualConnectionAndLayerNorm


class EncoderBlock(Module):
    """
    Core encoder block, composed of, from inputs to outputs:
    - multi-headed self-attention layer;
    - residual connection;
    - layer-normalization layer;
    - fully-connected (feed-forward) layer;
    - residual connection;
    - layer-normalization layer.
    """
    def __init__(self, feature_dimension: int,
                 self_multi_headed_attention_layer: Module,
                 fully_connected_layer: Module,
                 dropout_prob: float) -> None:
        super(EncoderBlock, self).__init__()
        self.feature_dimension = feature_dimension
        self.self_multi_headed_attention_layer = \
            self_multi_headed_attention_layer
        self.fully_connected_layer = fully_connected_layer
        self.residual_connection_blocks = get_clones(
            module_to_be_cloned=ResidualConnectionAndLayerNorm(
                feature_dimension=feature_dimension,
                dropout_prob=dropout_prob
            ),
            n_clones=2
        )

    def forward(self, src_features: Tensor, mask: Tensor) -> Tensor:
        """
        Forward propagation.
        """
        # self-attention, towards encoder token positions themselves, followed
        # by residual connection and layer normalization:
        src_features = self.residual_connection_blocks[0](
            src_features,
            lambda x: self.self_multi_headed_attention_layer(
                query_tokens=x,
                key_or_value_tokens=x,
                mask=mask
            )
        )
        # fully-connected (feed-forward) layer followed by residual connection
        # and layer normalization:
        return self.residual_connection_blocks[1](src_features,
                                                  self.fully_connected_layer)


class Encoder(Module):
    """
    Whole encoder, composed of repeated encoder blocks which do not share
    parameters.
    """
    def __init__(self, base_block, n_clones) -> None:
        super(Encoder, self).__init__()
        self.layers = get_clones(
            module_to_be_cloned=base_block,
            n_clones=n_clones
        )
        self.normalization_layer = LayerNorm(base_block.feature_dimension)
        # TODO: see TODO below

    def forward(self, src_features: Tensor, mask: Tensor) -> Tensor:
        """
        Forward propagation.
        """
        # forwarding inputs throught all encoder blocks:
        for layer in self.layers:
            src_features = layer(src_features=src_features, mask=mask)
        return self.normalization_layer(src_features)
        # TODO: understand why this last, additional normalization and why
        # it is not to be masked
