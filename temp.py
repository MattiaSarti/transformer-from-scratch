"""
Temp
"""


from transformer.transformer import Transformer


MAX_SEQUENCE_LENGTH = 20  # [number of tokens]

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

# initializing the model:
model = Transformer(
    src_vocabulary_dimension=10000,
    tgt_vocabulary_dimension=10000,
    n_encoder_blocks=2,  # 6,
    n_decoder_blocks=2,  # 6,
    representation_dimension=512,
    feedforward_dimension=2048,
    n_attention_heads=8,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
    dropout_prob=0.1
)

# training the model:
model.train_on_toy_copy_task(
    n_epochs=1,
    epoch_samples=2*2,
    mini_batch_size=2,
    label_smoothing_factor=0.0,
    learning_rate_n_warmup_steps=400,
    learning_rate_amplification_factor=1,
    gpu_if_possible=True
)
