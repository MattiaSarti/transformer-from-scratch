"""
Utilities for loading data.
"""


from torchtext.data import batch as torchtext_batch, Field, Iterator
from torchtext.datasets import IWSLT


class DatasetIterator(Iterator):
    """
    Iterator through mini-batches of the dataset, sorting them so as to
    cluster samples with a similar length in the same mini-batches, reducing
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
