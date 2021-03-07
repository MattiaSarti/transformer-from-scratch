"""
Utilities for loading data.
"""


from torchtext.data import Field, Iterator
from torchtext.data import batch as torchtext_batch
from torchtext.datasets import IWSLT

from transformer.training.preprocessing import Tokenizer


class DatasetIterator(Iterator):
    """
    Iterator through mini-batches of the dataset, sorting them so as to
    cluster samples with a similar lenght in the same mini-batches, reducing
    padding requirements to the minimum.
    """
    def create_batches(self):

        if self.train:

            def pool(data, random_shuffler) -> :
                # TODO: data types
                "."
                pass

            self.batches = pool(self.data(), self.random_shuffler)

        else:

            self.batches = []
            # for each mini-batch until all the dataset is covered:
            for mini_batch in torchtext_batch(
                    data=self.data(),
                    batch_size=self.batch_size,
                    batch_size_fn=self.batch_size_fn
                    ):
                # appending ...TODO: understand what
                self.batches.append(
                    sorted(mini_batch, key=self.sort_key)
                )


if __name__ == '__main__':
    
    # TODO: understand how to structure these instructions:
    # max_sequence_length: int, min_vocabulary_counts: int
    # self.max_sequence_length = max_sequence_length
    # self.min_vocabulary_counts = min_vocabulary_counts

    max_sequence_length = 100  # [number of tokens]
    min_vocabulary_counts = 2

    tokenizer = Tokenizer(src_language='de', tgt_language='en')

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
        exts=('.de', '.en'),
        fields=(src_data_handler, tgt_data_handler),
        # choosing only samples for which filter_pred(sample) is True,
        # corresponding to samples where both the source and the target
        # sequences are shorter or equal to the maximum allowed length:
        filter_pred=lambda x: (
            (len(vars(x)['src']) <= max_sequence_length) and
            (len(vars(x)['trg']) <= max_sequence_length)
            # TODO: adjust names of attributes ("MiniBatch" class ?)
        )
    )

    # building source and target dictionaries from already tokenized training
    # samples:
    src_data_handler.build_vocab(
        training_samples.src,
        # TODO: adjust name of attribute ("MiniBatch" class ?)
        min_freq=min_vocabulary_counts
    )
    tgt_data_handler.build_vocab(
        training_samples.trg,
        # TODO: adjust name of attribute ("MiniBatch" class ?)
        min_freq=min_vocabulary_counts
    )


# TODO: set seed for deterministic, reproducible results:
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     numpy.random.seed(worker_seed)
#     random.seed(worker_seed)

# DataLoader(
#     train_dataset,
#     batch_size=batch_size,
#     num_workers=num_workers,
#     worker_init_fn=seed_worker
# )
