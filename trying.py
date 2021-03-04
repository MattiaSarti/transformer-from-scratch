"""
Just trying.
"""


from transformer import Transformer


if __name__ == '__main__':

    model = Transformer(
        src_vocabulary_dimension=10,
        tgt_vocabulary_dimension=10,
        n_encoder_blocks=3,
        n_decoder_blocks=3,
        token_representation_dimension=12,
        feedforward_dimension=24,
        n_attention_heads=4,
        max_sequence_length=10,
        dropout_prob=0.1
    )

    model.train_on_toy_copy_task(
        epoch_samples=16,
        mini_batch_size=4,
        padding_token=0,
        label_smoothing_factor=0.1
    )
