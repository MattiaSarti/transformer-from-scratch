"""
Encoder architecture.
"""


from typing import NamedTuple

from torch import Tensor
from torch.nn import Module

from transformer.architecture.base import get_clones, LayerNorm,\
    ResidualConnectionAndLayerNorm


EncoderBlockBuildingBlocks = NamedTuple(
    'EncoderBuildingBlocks',
    [
        ('self_multi_head_attention_layer', Module),
        ('fully_connected_layer', Module)
    ]
)


class EncoderBlock(Module):
    """
    Core encoder block, composed of, from inputs to outputs:
    - multi-head self-attention layer;
    - residual connection;
    - layer-normalization layer;
    - fully-connected (feed-forward) layer;
    - residual connection;
    - layer-normalization layer.
    """

    def __init__(self, building_blocks: EncoderBlockBuildingBlocks,
                 feature_dimension: int, dropout_prob: float) -> None:
        super(EncoderBlock, self).__init__()
        self.feature_dimension = feature_dimension
        self.self_multi_head_attention_layer = (
            building_blocks.self_multi_head_attention_layer
        )
        self.fully_connected_layer = building_blocks.fully_connected_layer
        self.residual_connection_blocks = get_clones(
            module_to_be_cloned=ResidualConnectionAndLayerNorm(
                feature_dimension=feature_dimension,
                dropout_prob=dropout_prob
            ),
            n_clones=2
        )

    def forward(self, src_features: Tensor, src_mask: Tensor) -> Tensor:
        """
        Forward propagation.

        Tensor Shapes:

            Args:
                src_features: (batch size, sequence_length, n features)
                src_mask: (batch size, 1, sequence length)

            Returns:
                (batch size, sequence length, n. features)

        """
        # self-attention, towards encoder token positions themselves, followed
        # by residual connection and layer normalization:
        src_features = self.residual_connection_blocks[0](
            features=src_features,
            base_layer_call=lambda x: self.self_multi_head_attention_layer(
                query_tokens=x,
                key_or_value_tokens=x,
                mask=src_mask
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
        self.layer_blocks = get_clones(module_to_be_cloned=base_block,
                                       n_clones=n_clones)
        self.normalization_layer = LayerNorm(base_block.feature_dimension)

    def forward(self, src_features: Tensor, src_mask: Tensor) -> Tensor:
        """
        Forward propagation.

        Tensor Shapes:

            Args:
                src_features: (batch size, sequence_length, n. features)
                src_mask: (batch size, 1, sequence_length)

            Returns:
                (batch size, sequence length, n. features)

        """
        # forwarding inputs throught all encoder blocks:
        for layer_block in self.layer_blocks:
            src_features = layer_block(src_features=src_features,
                                       src_mask=src_mask)

        # a final, additional normalization is carried out:
        return self.normalization_layer(src_features)
