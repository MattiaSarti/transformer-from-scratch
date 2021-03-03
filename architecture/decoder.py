"""
Decoder architecture.
"""


from torch import Tensor
from torch.nn import Module

from .base import get_clones, LayerNorm, ResidualConnectionAndLayerNorm


class DecoderBlock(Module):
    """
    Core decoder block, composed of, from inputs to outputs:
    - multi-headed self-attention layer;
    - residual connection;
    - layer-normalization layer;
    - multi-headed source-attention layer;
    - residual connection;
    - layer-normalization layer;
    - fully-connected (feed-forward) layer;
    - residual connection;
    - layer-normalization layer.
    """
    def __init__(self, feature_dimension: int,
                 self_multi_headed_attention_layer: Module,
                 source_multi_headed_attention_layer: Module,
                 fully_connected_layer: Module, dropout_prob: float)\
            -> None:
        super(DecoderBlock, self).__init__()
        self.feature_dimension = feature_dimension
        self.self_multi_headed_attention_layer = \
            self_multi_headed_attention_layer
        self.source_multi_headed_attention_layer = \
            source_multi_headed_attention_layer
        self.fully_connected_layer = fully_connected_layer
        self.residual_connection_blocks = get_clones(
            module_to_be_cloned=ResidualConnectionAndLayerNorm(
                feature_dimension=feature_dimension,
                dropout_prob=dropout_prob
            ),
            n_clones=3
        )

    def forward(self, x: Tensor, src_encoded_tokens: Tensor,
                src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        # self-attention, towards decoder token positions themselves, followed
        # by residual connection and layer normalization:
        x = self.residual_connection_blocks[0](
            x,
            lambda x: self.self_multi_headed_attention_layer(
                query_tokens=x,
                key_or_value_tokens=x,
                mask=tgt_mask
            )
        )
        # source-attention, towards (final representations of) encoder token
        # positions, followed by residual connection and layer normalization:
        x = self.residual_connection_blocks[1](
            x,
            lambda x: self.source_multi_headed_attention_layer(
                query_tokens=x,
                key_or_value_tokens=src_encoded_tokens,
                mask=src_mask
            )
        )
        # fully-connected (feed-forward) layer followed by residual connection
        # and layer normalization:
        return self.residual_connection_blocks[2](x,
                                                  self.fully_connected_layer)


class Decoder(Module):
    """
    Whole decoder, composed of repeated decoder blocks which do not share
    parameters.
    """
    def __init__(self, base_block, n_clones) -> None:
        super(Decoder, self).__init__()
        self.layers = get_clones(
            module_to_be_cloned=base_block,
            n_clones=n_clones
        )
        self.normalization_layer = LayerNorm(base_block.feature_dimension)
        # TODO: see TODO below

    def forward(self, x: Tensor, src_encoded_tokens: Tensor,
                src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        # forwarding inputs throught all decoder blocks:
        for layer in self.layers:
            x = layer(x=x, src_encoded_tokens=src_encoded_tokens,
                      src_mask=src_mask, tgt_mask=tgt_mask)
        return self.normalization_layer(x)
        # TODO: understand why this last, additional normalization and why
        # it is not to be masked
