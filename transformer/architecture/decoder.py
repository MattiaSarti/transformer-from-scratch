"""
Decoder architecture.
"""


from typing import NamedTuple

from torch import Tensor
from torch.nn import Module

from transformer.architecture.base import get_clones, LayerNorm,\
    ResidualConnectionAndLayerNorm


DecoderBlockBuildingBlocks = NamedTuple(
    'DecoderBuildingBlocks',
    [
        ('self_multi_head_attention_layer', Module),
        ('source_multi_head_attention_layer', Module),
        ('fully_connected_layer', Module)
    ]
)


class DecoderBlock(Module):
    """
    Core decoder block, composed of, from inputs to outputs:
    - multi-head self-attention layer;
    - residual connection;
    - layer-normalization layer;
    - multi-head source-attention layer;
    - residual connection;
    - layer-normalization layer;
    - fully-connected (feed-forward) layer;
    - residual connection;
    - layer-normalization layer.
    """

    def __init__(self, building_blocks: DecoderBlockBuildingBlocks,
                 feature_dimension: int, dropout_prob: float) -> None:
        super(DecoderBlock, self).__init__()
        self.feature_dimension = feature_dimension
        self.self_multi_head_attention_layer = \
            building_blocks.self_multi_head_attention_layer
        self.source_multi_head_attention_layer = \
            building_blocks.source_multi_head_attention_layer
        self.fully_connected_layer = building_blocks.fully_connected_layer
        self.residual_connection_blocks = get_clones(
            module_to_be_cloned=ResidualConnectionAndLayerNorm(
                feature_dimension=feature_dimension,
                dropout_prob=dropout_prob
            ),
            n_clones=3
        )

    def forward(self, tgt_features: Tensor, src_encoded_tokens: Tensor,
                tgt_mask: Tensor, src_mask: Tensor) -> Tensor:
        """
        Forward propagation.

        Tensor Shapes:

            Args:
                tgt_features: (batch size, tgt. sequence length, n. features)
                src_encoded_tokens: (batch size, src. sequence length,
                    n. features)
                tgt_mask: (batch size, tgt. sequence length,
                    tgt. sequence length)
                src_mask: (batch size, 1, src. sequence length)

            Returns:
                (batch size, tgt. sequence_length, n. features)

        """
        # self-attention, towards decoder token positions themselves, followed
        # by residual connection and layer normalization:
        tgt_features = self.residual_connection_blocks[0](
            tgt_features,
            lambda x: self.self_multi_head_attention_layer(
                query_tokens=x,
                key_or_value_tokens=x,
                mask=tgt_mask
            )
        )
        # source-attention, towards (final representations of) encoder token
        # positions, followed by residual connection and layer normalization:
        tgt_features = self.residual_connection_blocks[1](
            tgt_features,
            lambda x: self.source_multi_head_attention_layer(
                query_tokens=x,
                key_or_value_tokens=src_encoded_tokens,
                mask=src_mask
            )
        )
        # fully-connected (feed-forward) layer followed by residual connection
        # and layer normalization:
        return self.residual_connection_blocks[2](tgt_features,
                                                  self.fully_connected_layer)


class Decoder(Module):
    """
    Whole decoder, composed of repeated decoder blocks which do not share
    parameters.
    """

    def __init__(self, base_block, n_clones) -> None:
        super(Decoder, self).__init__()
        self.layer_blocks = get_clones(module_to_be_cloned=base_block,
                                       n_clones=n_clones)
        self.normalization_layer = LayerNorm(base_block.feature_dimension)
        # TODO: see TODO below

    def forward(self, tgt_features: Tensor, src_encoded_tokens: Tensor,
                tgt_mask: Tensor, src_mask: Tensor) -> Tensor:
        """
        Forward propagation.

        Tensor Shapes:

            Args:
                tgt_features: (batch size, tgt. sequence length, n. features)
                src_encoded_tokens: (batch size, src. sequence length,
                    n. features)
                tgt_mask: (batch size, tgt. sequence length,
                    tgt. sequence length)
                src_mask: (batch size, 1, tgt. sequence length)

            Returns:
                (batch size, tgt. sequence_length, n features)

        """
        # forwarding inputs throught all decoder blocks:
        for layer_block in self.layer_blocks:
            tgt_features = layer_block(tgt_features=tgt_features,
                                       src_encoded_tokens=src_encoded_tokens,
                                       src_mask=src_mask, tgt_mask=tgt_mask)
        return self.normalization_layer(tgt_features)
        # TODO: understand why this last, additional normalization and why
        # it is not to be masked
