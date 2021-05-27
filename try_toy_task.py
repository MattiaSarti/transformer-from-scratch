"""
Trying the model on a toy task: copying the source sentence as the target one,
printing examples before and after training to evaluate the different behavior
achieved.
"""


from typing import Tuple

from torch import (  # pylint: disable=no-name-in-module
    cat as torch_cat, long as torch_long, ones as torch_ones, unsqueeze as
    torch_unsqueeze, tensor, Tensor, zeros as torch_zeros
)

from transformer.training_and_inference.reproducibility import\
    make_results_reproducible
from transformer.transformer import Transformer


def get_sequence_from_user(max_sequence_length: int) -> Tuple[Tensor, Tensor]:
    """
    Ask the user to enter a sequence of token ids and convert it to source
    token tensor and source mask tensor for feeding the model.
    """
    enter_message = (
        "\nEnter the desired source sequence token ids separated by spaces: "
    )

    # asking for user input and splitting it into a sequence of token ids:
    src_seq = list(map(int, input(enter_message).split()))
    n_tokens = len(src_seq)

    if n_tokens > max_sequence_length:
        # truncating the sequence if its length is higher than allowed:
        n_tokens = max_sequence_length
        src_seq = src_seq[: max_sequence_length]

    # padding the sequence if its length is lower than the maximum one and
    # converting it to the right format:
    src_seq = torch_cat(
        (
            tensor(src_seq, dtype=torch_long),  # noqa: E501 pylint: disable=not-callable
            torch_zeros((max_sequence_length - n_tokens),
                        dtype=torch_long)
        ),
        dim=-1
    )
    src_seq = torch_unsqueeze(input=src_seq, dim=0)

    # creating the sequence mask based on the padding done:
    src_seq_mask = torch_cat(
        (
            torch_ones((1, 1, n_tokens), dtype=torch_long),
            torch_zeros((1, 1, max_sequence_length - n_tokens),
                        dtype=torch_long)
        ),
        dim=-1
    )

    return src_seq, src_seq_mask


def print_src_vs_tgt(src_seq: Tensor, tgt_seq_prediction: Tensor)\
         -> None:
    """
    Print source sequence and predicted sequence to standard output.
    """
    print('\n')
    print("Source sequence:", src_seq)
    print("-> Predicted target sequence:", tgt_seq_prediction)
    print('\n')


if __name__ == '__main__':

    MAX_SEQUENCE_LENGTH = 10  # [number of tokens]

    # for reproducible results:
    make_results_reproducible()

    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    # initializing the model:
    model = Transformer(
        src_vocabulary_dimension=11,
        tgt_vocabulary_dimension=11,
        n_encoder_blocks=6,
        n_decoder_blocks=6,
        representation_dimension=512,
        feedforward_dimension=2048,
        n_attention_heads=8,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        dropout_prob=0.1
    )

    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    # evaluating a single prediction before training:
    src_sequence = torch_unsqueeze(
        input=tensor(list(range(1, MAX_SEQUENCE_LENGTH + 1))),  # noqa: E501 pylint: disable=not-callable
        # input=tensor([2] * MAX_SEQUENCE_LENGTH),  # TODO  # noqa: E501 pylint: disable=not-callable
        dim=0
    )
    src_sequence_mask = torch_ones((1, 1, MAX_SEQUENCE_LENGTH))
    tgt_sequence_prediction = model.predict(
        src_sequences=src_sequence,
        src_masks=src_sequence_mask,
        tgt_bos_token=1,
        decoding_method='greedy',
        gpu_if_possible=True
    )
    print_src_vs_tgt(
        src_seq=src_sequence,
        tgt_seq_prediction=tgt_sequence_prediction
    )

    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    # training the model:
    model.train_on_toy_copy_task(
        n_epochs=10,
        samples_per_epoch=30*20,
        mini_batch_size=30,
        label_smoothing_factor=0.0,
        learning_rate_n_warmup_steps=400,
        learning_rate_amplification_factor=1,
        gpu_if_possible=True
    )

    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    # evaluating the same, single prediction after training:
    tgt_sequence_prediction = model.predict(
        src_sequences=src_sequence,
        src_masks=src_sequence_mask,
        tgt_bos_token=1,
        decoding_method='greedy',
        gpu_if_possible=True
    )
    print_src_vs_tgt(
        src_seq=src_sequence,
        tgt_seq_prediction=tgt_sequence_prediction
    )

    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    print('~'*60 + '\n' + '~'*60)
    while True:

        # asking for user input and extracting the sequence ids and its mask:
        src_sequence, src_sequence_mask = get_sequence_from_user(
            max_sequence_length=MAX_SEQUENCE_LENGTH
        )

        # evaluating the respective prediction:
        tgt_sequence_prediction = model.predict(
            src_sequences=src_sequence,
            src_masks=src_sequence_mask,
            tgt_bos_token=1,
            decoding_method='greedy',
            gpu_if_possible=True
        )
        print_src_vs_tgt(
            src_seq=src_sequence,
            tgt_seq_prediction=tgt_sequence_prediction
        )
