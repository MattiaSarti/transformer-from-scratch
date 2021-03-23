"""
Trying the model on an IWSLT 2016 TED Talks dataset: a { German -> English }
translation task, printing examples before and after training to evaluate the
different behavior achieved.
"""


from random import seed as random_seed

from numpy.random import seed as numpy_seed
from torch import manual_seed

from .transformer.transformer import Transformer


if __name__ == '__main__':

    # for reproducible results:
    _ = manual_seed(0)
    _ = numpy_seed(0)
    random_seed(0)

    # iterators for the training and validation sets:
    ...

    # the source and target vocabulary sizes are imposed by the tokenizer:
    src_vocabulary_dimension = len(src_data_handler.vocab)
    tgt_vocabulary_dimension = len(tgt_data_handler.vocab)

    max_sequence_length = 100  # [number of tokens]

    model = Transformer(
        src_vocabulary_dimension=src_vocabulary_dimension,
        tgt_vocabulary_dimension=tgt_vocabulary_dimension,
        n_encoder_blocks=6,
        n_decoder_blocks=6,
        representation_dimension=512,
        feedforward_dimension=2048,
        n_attention_heads=8,
        max_sequence_length=max_sequence_length,
        dropout_prob=0.1
    )
