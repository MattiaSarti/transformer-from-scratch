"""
Trying the model on a toy task: copying the source sentence as the target one,
printing examples before and after training to evaluate the different behavior
achieved.
"""


from random import seed as random_seed
from typing import Tuple

from numpy.random import seed as numpy_seed
from torch import manual_seed, tensor, Tensor
from torch import cat as torch_cat, long as torch_long, ones as torch_ones,\
    unsqueeze as torch_unsqueeze, zeros as torch_zeros

from transformer.transformer import Transformer


def get_sequence_from_user() -> Tuple[Tensor, Tensor]:
    """
    Ask the user to enter a sequence of token ids and convert it to source
    token tensor and source mask tensor for feeding the model.
    """
    enter_message = "\nEnter the desired source sequence token ids separated"\
        " by spaces: "

    # asking for user input and splitting it into a sequence of token ids:
    src_sequence = [x for x in map(int, input(enter_message).split())]
    n_tokens = len(src_sequence)

    if n_tokens > max_sequence_length:
        # truncating the sequence if its length is higher than allowed:
        n_tokens = max_sequence_length
        src_sequence = src_sequence[: max_sequence_length]

    # padding the sequence if its length is lower than the maximum one and
    # converting it to the right format:
    src_sequence = torch_cat(
        (
            tensor(src_sequence, dtype=torch_long),
            torch_zeros((max_sequence_length - n_tokens),
                        dtype=torch_long)
        ),
        dim=-1
    )
    src_sequence = torch_unsqueeze(input=src_sequence, dim=0)

    # creating the sequence mask based on the padding done:
    src_sequence_mask = torch_cat(
        (
            torch_ones((1, 1, n_tokens), dtype=torch_long),
            torch_zeros((1, 1, max_sequence_length - n_tokens),
                        dtype=torch_long)
        ),
        dim=-1
    )

    return src_sequence, src_sequence_mask


def print_src_vs_tgt(src_sequence: Tensor, tgt_sequence_prediction: Tensor)\
         -> None:
    """
    Print source and predicted sequences.
    """
    print('\n')
    print("Source sequence:", src_sequence)
    print("-> Predicted target sequence:", tgt_sequence_prediction)
    print('\n')
    
if __name__ == '__main__':

    # for reproducible results:
    _ = manual_seed(0)
    _ = numpy_seed(0)
    random_seed(0)

    max_sequence_length = 10  # [number of tokens]

    # initializing the model:
    model = Transformer(
        src_vocabulary_dimension=10000,#11,
        tgt_vocabulary_dimension=10000,#11,
        n_encoder_blocks=6,
        n_decoder_blocks=6,
        representation_dimension=512,
        feedforward_dimension=2048,
        n_attention_heads=8,
        max_sequence_length=max_sequence_length,
        dropout_prob=0.1
    )

    # evaluating a single prediction before training:

    src_sequence = tensor([x for x in range(1, max_sequence_length + 1)])
    # src_sequence = tensor([2] * max_sequence_length)  # TODO
    src_sequence = torch_unsqueeze(input=src_sequence, dim=0)
    src_sequence_mask = torch_ones((1, 1, max_sequence_length))

    tgt_sequence_prediction = model.predict(
        src_sequences=src_sequence,
        src_masks=src_sequence_mask,
        tgt_bos_token=1,
        decoding_method='greedy'
    )

    print_src_vs_tgt(
        src_sequence=src_sequence,
        tgt_sequence_prediction=tgt_sequence_prediction
    )

    # training the model:

    model.train_on_toy_copy_task(
        n_epochs=100,
        epoch_samples=30*20,
        mini_batch_size=30,
        label_smoothing_factor=0.0,
        learning_rate_n_warmup_steps=400,
        learning_rate_amplification_factor=1
    )

    # evaluating the same, single prediction after training:

    tgt_sequence_prediction = model.predict(
        src_sequences=src_sequence,
        src_masks=src_sequence_mask,
        tgt_bos_token=1,
        decoding_method='greedy'
    )

    print_src_vs_tgt(
        src_sequence=src_sequence,
        tgt_sequence_prediction=tgt_sequence_prediction
    )

    print('~'*60 + '\n' + '~'*60)

    while True:

        # asking for user input and extracting the sequence ids and its mask:
        src_sequence, src_sequence_mask = get_sequence_from_user()

        tgt_sequence_prediction = model.predict(
            src_sequences=src_sequence,
            src_masks=src_sequence_mask,
            tgt_bos_token=1,
            decoding_method='greedy'
        )

        print_src_vs_tgt(
            src_sequence=src_sequence,
            tgt_sequence_prediction=tgt_sequence_prediction
        )
