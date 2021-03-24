from random import seed as random_seed
from typing import Tuple

from numpy.random import seed as numpy_seed
from torch import manual_seed as torch_manual_seed, tensor, Tensor
from torch import cat as torch_cat, long as torch_long, ones as torch_ones,\
    unsqueeze as torch_unsqueeze, zeros as torch_zeros

from transformer.transformer import Transformer


max_sequence_length = 100  # [number of tokens]

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

# initializing the model:
model = Transformer(
    src_vocabulary_dimension=10000,
    tgt_vocabulary_dimension=10000,
    n_encoder_blocks=6,
    n_decoder_blocks=6,
    representation_dimension=512,
    feedforward_dimension=2048,
    n_attention_heads=8,
    max_sequence_length=max_sequence_length,
    dropout_prob=0.1
)

# training the model:
model.train_on_toy_copy_task(
    n_epochs=1,
    epoch_samples=30*2,
    mini_batch_size=30,
    label_smoothing_factor=0.0,
    learning_rate_n_warmup_steps=400,
    learning_rate_amplification_factor=1
)