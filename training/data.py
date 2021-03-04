"""
Utilities for loading data.
"""


from typing import List

from spacy import load as spacy_load
from torchtext.data import Field, Iterator
from torchtext.data import batch as torchtext_batch
from torchtext.datasets import IWSLT


class DatasetIterator(Iterator):
    """
    Iterator through mini-batches of the dataset, sorting them so as to
    cluster samples with a similar lenght in the same mini-batches, reducing
    padding requirements to the minimum.
    """
    def create_batches(self):

        if self.train:

            pass

        else:

            self.batches = []
            # for each mini-batch untill all the dataset is covered:
            for mini_batch in torchtext_batch(self.data(), self.batch_size,
                                              self.batch_size_fn):
                # appending 
                self.batches.append(
                    sorted(mini_batch, key=self.sort_key)
                )


class Tokenizer:
    """
    Source and target tokenizers.
    """
    def __init__(self, src_language: str = 'de', tgt_language: str = 'en')\
            -> None:
        super(Tokenizer, self).__init__()

        self.src_language_model = spacy_load(src_language)
        self.tgt_language_model = spacy_load(tgt_language)

        self.bos_token = "<s>"  # beginning of sequence
        self.eos_token = "</s>"  # end of sequence
        self.unk_token = "<unk>"  # unknown
        self.padding_token = "<blank>"
        

    def tokenize_src(self, text: str) -> List[str]:
        "Tokenize source language text."
        return [
            token for token in self.src_language_model.tokenizer(text)
        ]

    def tokenize_tgt(self, text: str) -> List[str]:
        "Tokenize target language text."
        return [
            token for token in self.tgt_language_model.tokenizer(text)
        ]


# TODO: understand how to structure these instructions:
# max_sequence_length: int, min_vocabulary_counts: int
# self.max_sequence_length = max_sequence_length
# self.min_vocabulary_counts = min_vocabulary_counts
max_sequence_length = 100
min_vocabulary_counts = 2

tokenizer = Tokenizer()

# handlers for converting raw text into tokenized tensors:
src_data_handler = Field(
    tokenize=tokenizer.tokenize_src,
    init_token=None,  # not required for source tokens
    eos_token=None,  # not required for source tokens
    pad_token=tokenizer.padding_token,
    unk_token=tokenizer.unk_token,
)
tgt_data_handler = Field(
    tokenize=tokenizer.tokenize_tgt,
    init_token=tokenizer.bos_token,
    eos_token=tokenizer.eos_token,
    pad_token=tokenizer.padding_token,
    unk_token=tokenizer.unk_token
)

# loading the samples while splitting them among training, validation and
# test sets:
training_samples, val_samples, test_samples = IWSLT.splits(
    exts=('.de', '.en'),  # extensions to paths
    fields=(src_data_handler, tgt_data_handler),  # tensor format converters
    # choosing only samples for which filter_pred(sample) is True,
    # corresponding to samples where both the source and the target
    # sequences are shorter or equal to the maximum allowed length:
    filter_pred=lambda x: (
        (len(vars(x)['src']) <= max_sequence_length) and
        (len(vars(x)['trg']) <= max_sequence_length)
    )
)

# building source and target dictionaries from already tokenized training
# samples:
src_data_handler.build_vocsb(
    training_samples.src,
    min_freq=min_vocabulary_counts
)
tgt_data_handler.build_vocsb(
    training_samples.trg,
    min_freq=min_vocabulary_counts
)
