"""
Utilities for loading data.
"""


from typing import Generator, Optional

from numpy import int64 as numpy_int64
from numpy.random import randint
from torch import from_numpy, Tensor  # noqa: E501 pylint: disable=E0611
from torch.cuda import is_available as cuda_is_available
# from torchtext.data import batch as torchtext_batch, Field, Iterator
# from torchtext.datasets import IWSLT

from transformer.architecture.attention import allowed_positions_to_attend
# from transformer.training_and_inference.preprocessing import Tokenizer


# NOTE: in the original implementation, this was a class while it shouldn't
# have been, it just stores data after processing them during initialization,
# which is a NamedTuple in Pyhton:
class MiniBatch:
    """
    Mini-batch of samples.
    """

    @staticmethod
    def build_mask(tgt_tokens: Tensor, padding_token: int) -> Tensor:
        """
        Build masks of target positions allowed to be attended by the decoder,
        position by position.
        """
        tgt_mask = (tgt_tokens != padding_token).unsqueeze(dim=-2)
        tgt_mask = tgt_mask & (
            allowed_positions_to_attend(
                n_positions=tgt_tokens.size(-1)
            ).type_as(tgt_mask)
        )
        # NOTE: the original implementation had '&', which is the bit-wise
        # AND, in place of 'and', which is the logical AND... why? wasn't it
        # wrong?
        return tgt_mask

    def __init__(self, src_tokens: Tensor, padding_token: int,
                 tgt_tokens: Optional[Tensor] = None) -> None:
        # source inputs:
        self.src_tokens = src_tokens
        # all source positions are allowed to be attended, both by the
        # encoder and by decoder:
        self.src_mask = (src_tokens != padding_token).unsqueeze(dim=-2)
        # when target outputs specified:
        if tgt_tokens is not None:
            self.tgt_input_tokens = tgt_tokens[:, :-1]  # excluding </s> token
            self.tgt_expected_tokens = tgt_tokens[:, 1:]  # excluding <s> token
            self.actual_n_target_tokens = \
                (self.tgt_expected_tokens != padding_token).data.sum().item()
            # only target positions up to the current position are allowed to
            # be attended by the decoder, for each position:
            self.tgt_mask = self.build_mask(self.tgt_input_tokens,
                                            padding_token=padding_token)
    # NOTE: understand why shapes of tgt masks are different from src masks


# class MiniBatchHandler:
#     """
#     TODO
#     """

#     def __init__(self, max_n_src_tokens_in_mini_batch: int,
#                  max_n_tgt_tokens_in_mini_batch: int) -> None:
#         self.max_n_src_tokens_in_mini_batch = max_n_src_tokens_in_mini_batch
#         self.max_n_tgt_tokens_in_mini_batch = max_n_tgt_tokens_in_mini_batch

#     def get_current_mini_batch_size(self, new, count: int):
#         # TODO: add data type & understand why they add an unused, additional
#         # argument called 'sofar'
#         """
#         TODO
#         """
#         # resetting initial values when starting a new mini-batch size
#         # monitoring (during construction):
#         if count == 1:
#             self.max_n_src_tokens_in_mini_batch = 0
#             self.max_n_tgt_tokens_in_mini_batch = 0
#         # :
#         self.max_n_src_tokens_in_mini_batch = max(
#             self.max_n_src_tokens_in_mini_batch,
#             len()
#         )
#         self.max_n_tgt_tokens_in_mini_batch = max(
#             self.max_n_tgt_tokens_in_mini_batch,
#             len()
#         )
#         # TODO:
#         src_tokens = count * self.max_n_src_tokens_in_mini_batch
#         tgt_tokens = count * self.max_n_tgt_tokens_in_mini_batch
#         return max(src_tokens, tgt_tokens)


# class DatasetIterator(Iterator):
#     """
#     Iterator through mini-batches of the dataset, sorting them so as to
#     cluster samples with a similar length in the same mini-batches, reducing
#     padding requirements to the minimum.
#     """

#     def create_batches(self):

#         if self.train:

#             def pool(data, random_shuffler) -> :
#                 # TODO: data types
#                 "."
#                 pass

#             self.batches = pool(self.data(), self.random_shuffler)

#         else:

#             self.batches = []
#             # for each mini-batch until all the dataset is covered:
#             for mini_batch in torchtext_batch(
#                     data=self.data(),
#                     batch_size=self.batch_size,
#                     batch_size_fn=self.batch_size_fn
#                     ):
#                 # appending ...TODO: understand what
#                 self.batches.append(
#                     sorted(mini_batch, key=self.sort_key)
#                 )


def dataset_builder_copy_task(sequence_length: int, vocabulary_size: int,
                              mini_batch_size: int, n_mini_batches: int,
                              gpu_if_possible: bool = True)\
        -> Generator[MiniBatch, None, None]:
    """
    Build generator yielding dummy samples and labels for a toy source-target
    copy task.
    """
    for _ in range(n_mini_batches):

        # random token indices, excluding 0 because assumed to represent the
        # padding token:
        samples = from_numpy(
            randint(
                low=1,
                high=vocabulary_size,
                # same length for all sequences, in this toy task:
                size=(mini_batch_size, sequence_length),
                dtype=numpy_int64
            )
        )

        # selecting the device handling computations:
        if gpu_if_possible:
            # employing a GPU if possible:
            device = 'cuda:0' if cuda_is_available() else 'cpu'
        else:
            device = 'cpu'

        # moving samples to such device:
        samples = samples.to(device)

        # assuming all sequences start with the same token, an hypothetical
        # <s> token that can also be found in other positions of the sequences
        # in this toy task:
        samples[:, 0] = 1

        # yielding mini-batch made of identical source and target samples
        # (i.e. labels equal samples):
        yield MiniBatch(
            src_tokens=samples.detach().clone(),  # graph-detached, deep copy
            tgt_tokens=samples.detach().clone(),  # graph-detached, deep copy
            padding_token=0  # as assumed above
        )


# def dataset_builder_IWSLT_task(max_sequence_length: int) -> None:
#     # TODO: understand returned data type -> Tuple[, , int]:
#     """
#     .
#     """
#     min_vocabulary_counts = 2

#     tokenizer = Tokenizer(src_language='de', tgt_language='en')

#     # handlers for converting raw text into tokenized tensors:
#     src_data_handler = Field(
#         tokenize=tokenizer.tokenize_src,
#         init_token=None,  # not required for source tokens
#         eos_token=None,  # not required for source tokens
#         pad_token=tokenizer.padding_token,
#         unk_token=tokenizer.unk_token,
#     )
#     tgt_data_handler = Field(
#         tokenize=tokenizer.tokenize_tgt,
#         init_token=tokenizer.bos_token,
#         eos_token=tokenizer.eos_token,
#         pad_token=tokenizer.padding_token,
#         unk_token=tokenizer.unk_token
#     )

#     # loading the samples while splitting them among training, validation and
#     # test sets:
#     training_samples, validation_samples, test_samples = IWSLT.splits(
#         exts=('.de', '.en'),
#         fields=(src_data_handler, tgt_data_handler),
#         # choosing only samples for which filter_pred(sample) is True,
#         # corresponding to samples where both the source and the target
#         # sequences are shorter or equal to the maximum allowed length:
#         filter_pred=lambda x: (
#             (len(vars(x)['src']) <= max_sequence_length) and
#             (len(vars(x)['trg']) <= max_sequence_length)
#             # TODO: adjust names of attributes ("MiniBatch" class ?)
#         )
#     )

#     # building source and target dictionaries from already tokenized training
#     # samples:
#     src_data_handler.build_vocab(
#         training_samples.src_tokens,
#         # TODO: check name of attribute ("MiniBatch" class ?)
#         min_freq=min_vocabulary_counts
#     )
#     tgt_data_handler.build_vocab(
#         training_samples.tgt_input_tokens,
#         # TODO: check name of attribute ("MiniBatch" class ?)
#         min_freq=min_vocabulary_counts
#     )

#     # padding token id as imposed by the tokenizer:
#     padding_token = tgt_data_handler.vocab.stoi["<blank>"]

#     # ordering samples in mini-batches so as to group samples with similar
#     # lengths in the same mini-batches, minimizing padding requirements
#     # within mini-batches:
#     training_iterator = DatasetIterator(
#         training_samples,
#         batch_size=BATCH_SIZE,
#         device=0,
#         repeat=False,
#         sort_key=lambda x: (len(x.src), len(x.trg)),
#         batch_size_fn=batch_size_fn,
#         train=True
#     )
#     training_iterator = (
#         rebatch(padding_token=padding_token, mini_batch=mini_batch) for
#             mini_batch in training_iterator
#     )
#     validation_iterator = DatasetIterator(
#         validation_samples,
#         batch_size=BATCH_SIZE,
#         device=0,
#         repeat=False,
#         sort_key=lambda x: (len(x.src), len(x.trg)),
#         batch_size_fn=batch_size_fn,
#         train=False
#     )
#     validation_iterator = (
#         rebatch(padding_token=padding_token, mini_batch=mini_batch) for
#             mini_batch in validation_iterator
#     )

#     return training_iterator, validation_iterator, padding_token

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
# def rebatch(padding_token: int, mini_batch: MiniBatch) -> MiniBatch:
#     pass
