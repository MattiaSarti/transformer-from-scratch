"""
Trying the model on an IWSLT 2016 TED Talks dataset: the { German -> English }
translation task, printing examples before and after training to evaluate the
different behavior achieved.
"""


from random import seed as random_seed

from numpy.random import seed as numpy_seed
from torch import manual_seed

from transformer.training_and_inference.reproducibility import\
    make_results_reproducible
from .transformer.transformer import Transformer


if __name__ == '__main__':

    # for reproducible results:
    make_results_reproducible()

    # iterators for the training and validation sets:
    ...

    # the source and target vocabulary sizes are imposed by the tokenizer:
    src_vocabulary_dimension = len(src_data_handler.vocab)
    tgt_vocabulary_dimension = len(tgt_data_handler.vocab)

    max_sequence_length = 100  # [number of tokens]

    model = Transformer(
        ...
    )
