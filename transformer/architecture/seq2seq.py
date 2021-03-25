"""
Sequence-to-sequence encoder-decoder architecture scaffold.
"""


from typing import NamedTuple

from torch import Tensor
from torch.nn import Module


Seq2SeqBuildingBlocks = NamedTuple(
    'Seq2SeqBuildingBlocks',
    [
        ('encoder', Module),
        ('decoder', Module),
        ('src_embedder', Module),
        ('tgt_embedder', Module),
        ('log_softmax_layer', Module)
    ]
)


class Seq2Seq(Module):
    """
    Base architecture for encoder-decoder sequence-to-sequence Transformer
    models.
    """

    def __init__(self, building_blocks: Seq2SeqBuildingBlocks) -> None:

        super(Seq2Seq, self).__init__()
        self.encoder = building_blocks.encoder
        self.decoder = building_blocks.decoder
        self.src_embedder = building_blocks.src_embedder
        self.tgt_embedder = building_blocks.tgt_embedder
        self.log_softmax_layer = building_blocks.log_softmax_layer

    def forward(self, src_tokens: Tensor, tgt_tokens: Tensor, src_mask:
                Tensor, tgt_mask: Tensor) -> Tensor:
        """
        Forward propagation.

        Tensor Shapes:

            Args:
                src_tokens: (batch size, sequence length)
                tgt_tokens: (batch size, sequence length - 1)
                src_mask: (batch size, 1, sequence length)
                tgt_mask: (batch size, sequence length - 1,
                    sequence length - 1)

            Returns:
                (batch size, sequence length - 1, n. features)

        """
        return self.decode(
            src_encoded_tokens=self.encode(
                src_tokens=src_tokens,
                src_mask=src_mask
            ),
            src_mask=src_mask,
            tgt_tokens=tgt_tokens,
            tgt_mask=tgt_mask
        )

    def encode(self, src_tokens: Tensor, src_mask: Tensor) -> Tensor:
        """
        Encode the input source tokens and their masks to produce their final
        encoder representations (i.e. the outputs of the last encoder layer).
        """
        return self.encoder(
            src_features=self.src_embedder(src_tokens),
            src_mask=src_mask
        )

    def decode(self, src_encoded_tokens: Tensor, src_mask: Tensor,
               tgt_tokens: Tensor, tgt_mask: Tensor) -> Tensor:
        """
        Decode the encoded source token representations (i.e. the outputs of
        the last encoder layer) and their masks, together with the already
        decoded output tokens (i.e. the already computed outputs of the last
        decoder layer) and their masks, predicting the next target token.
        """
        return self.decoder(
            tgt_features=self.tgt_embedder(tgt_tokens),
            src_encoded_tokens=src_encoded_tokens,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
