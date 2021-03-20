"""
Just trying.
"""


from random import seed as random_seed

from numpy.random import seed as numpy_seed
from torch import manual_seed, tensor
from torch import cat as torch_cat
from torch import long as torch_long
from torch import ones as torch_ones
from torch import zeros as torch_zeros

from transformer.transformer import Transformer


if __name__ == '__main__':

    # for reproducible results:
    _ = manual_seed(0)
    _ = numpy_seed(0)
    _ = random_seed(0)

    max_sequence_length = 10

    model = Transformer(
        src_vocabulary_dimension=11,
        tgt_vocabulary_dimension=11,
        n_encoder_blocks=6,
        n_decoder_blocks=6,
        token_representation_dimension=512,
        feedforward_dimension=2048,
        n_attention_heads=8,
        max_sequence_length=max_sequence_length,
        dropout_prob=0.1
    )

    src_sequence = tensor([x for x in range(1, 11)])
    src_sequence_mask = torch_ones((1, 1, max_sequence_length))

    tgt_sequence_prediction = model.predict(
        src_sequences=src_sequence,
        src_masks=src_sequence_mask,
        tgt_bos_token=1,
        decoding_method='greedy'
    )

    print('\n')
    print("Source sequence:", src_sequence)
    print("-> Predicted target sequence:", tgt_sequence_prediction)
    print('\n')

    model.train_on_toy_copy_task(
        n_epochs=100,
        epoch_samples=30*20,
        mini_batch_size=30,
        label_smoothing_factor=0.0,
        learning_rate_n_warmup_steps=400,
        learning_rate_amplification_factor=1
    )

    tgt_sequence_prediction = model.predict(
        src_sequences=src_sequence,
        src_masks=src_sequence_mask,
        tgt_bos_token=1,
        decoding_method='greedy'
    )

    print('\n')
    print("Source sequence:", src_sequence)
    print("-> Predicted target sequence:", tgt_sequence_prediction)
    print('\n')

    print('~'*60 + '\n' + '~'*60)
    enter_message = "\nEnter the desired source sequence token ids separated"\
        " by spaces: "

    while True:

        src_sequence = [x for x in map(int, input(enter_message).split())]
        n_tokens = len(src_sequence)
        if n_tokens > max_sequence_length:
            n_tokens = max_sequence_length
            src_sequence = src_sequence[: max_sequence_length]

        src_sequence = torch_cat(
            (
                tensor(src_sequence, dtype=torch_long),
                torch_zeros((max_sequence_length - n_tokens),
                            dtype=torch_long)
            ),
            dim=-1
        )
        src_sequence_mask = torch_cat(
            (
                torch_ones((1, 1, n_tokens), dtype=torch_long),
                torch_zeros((1, 1, max_sequence_length - n_tokens),
                            dtype=torch_long)
            ),
            dim=-1
        )

        tgt_sequence_prediction = model.predict(
            src_sequences=src_sequence,
            src_masks=src_sequence_mask,
            tgt_bos_token=1,
            decoding_method='greedy'
        )

        print('\n')
        print("\tSource sequence:", src_sequence)
        print("\t-> Predicted target sequence:", tgt_sequence_prediction)
        print('\n')
