# How I Reimplemented the Original Transformer from Scratch

An exercise to learn about Transformers: a PyTorch implementation of the original model [[1](https://github.com/MattiaSarti/transformer-from-scratch#references)] that followed an already existing one [[2](https://github.com/MattiaSarti/transformer-from-scratch#references)] - with some small differences, though - to prove myself I could build a Transformer from scratch and train it.\
After gaining theoretical intuition behind the architecture logic [[3](https://github.com/MattiaSarti/transformer-from-scratch#references)], I went through every line of code [[2](https://github.com/MattiaSarti/transformer-from-scratch#references)] and the respective libraries' docs, reproducing such steps while customizing, testing and abundantly documenting them.

## Purpose
I was interested in understanding the details of the [architecture implementation](https://github.com/MattiaSarti/transformer-from-scratch/tree/main/transformer/architecture) - to then propose a modification of the position encoding mechanism, as done in [this other repository](https://github.com/MattiaSarti/rethinking-position-encoding-in-transformers) - but following [[2](https://github.com/MattiaSarti/transformer-from-scratch#references)] I also [trained and evaluated](https://github.com/MattiaSarti/transformer-from-scratch/tree/main/transformer/training_and_inference) the model on a toy source-to-target copy task to ensure its proper functioning. An example of resulting translations follows, reproducible by running ```python try_toy_task.py``` in the root directory of the repository.\
```
A Toy Source-to-Target Copy Task

Evaluating a single prediction before training:
Source sequence:
         1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10
Predicted sequence:
         1 -> 2 -> 2 -> 2 -> 2 -> 2 -> 2 -> 2 -> 2 -> 2

Training the model:
------------------------------------------------------------
Epoch 1/10
Mini-batches done: 1 - Loss for the current mini-batch: 3.0322 - Average speed [tokens/s]: 1588.2
Average Loss per Token: 1.829
------------------------------------------------------------
Epoch 2/10
Mini-batches done: 1 - Loss for the current mini-batch: 1.8724 - Average speed [tokens/s]: 2327.6
Average Loss per Token: 1.673
------------------------------------------------------------
Epoch 3/10
Mini-batches done: 1 - Loss for the current mini-batch: 1.7440 - Average speed [tokens/s]: 3033.0
Average Loss per Token: 1.665
------------------------------------------------------------
Epoch 4/10
Mini-batches done: 1 - Loss for the current mini-batch: 1.7157 - Average speed [tokens/s]: 3000.0
Average Loss per Token: 1.551
------------------------------------------------------------
Epoch 5/10
Mini-batches done: 1 - Loss for the current mini-batch: 1.5440 - Average speed [tokens/s]: 3000.0
Average Loss per Token: 1.195
------------------------------------------------------------
Epoch 6/10
Mini-batches done: 1 - Loss for the current mini-batch: 1.2815 - Average speed [tokens/s]: 3034.2
Average Loss per Token: 0.959
------------------------------------------------------------
Epoch 7/10
Mini-batches done: 1 - Loss for the current mini-batch: 0.9736 - Average speed [tokens/s]: 2967.2
Average Loss per Token: 0.921
------------------------------------------------------------
Epoch 8/10
Mini-batches done: 1 - Loss for the current mini-batch: 0.8368 - Average speed [tokens/s]: 3000.1
Average Loss per Token: 0.520
------------------------------------------------------------
Epoch 9/10
Mini-batches done: 1 - Loss for the current mini-batch: 0.6759 - Average speed [tokens/s]: 3000.1
Average Loss per Token: 0.593
------------------------------------------------------------
Epoch 10/10
Mini-batches done: 1 - Loss for the current mini-batch: 0.6947 - Average speed [tokens/s]: 3033.8
Average Loss per Token: 0.350
------------------------------------------------------------

Evaluating the same prediction after training:
Source sequence:
         1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10
Predicted sequence:
         1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 8 -> 9 -> 10 -> 9

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enter the desired source sequence token ids separated by spaces: 2 5 1 8 4 9
Source sequence:
         2 -> 5 -> 1 -> 8 -> 4 -> 9 -> 0 -> 0 -> 0 -> 0
Predicted sequence:
         1 -> 2 -> 1 -> 8 -> 4 -> 9 -> 4 -> 9 -> 9 -> 9
```

## Requirements
Python 3.8.0 + NVIDIA driver + CUDA 11.2 + Cudnn + pip install -r requirements.txt

## Tests
✓ Unit Tests: passed - to reproduce the outcome, run ```python run_tests.py``` in the repository root directory\
✓ Code Coverage: *<...coming soon...>* %

## Code Style
✓ Pylint-compliant (2.5.3)\
✓ Flake8-compliant (3.8.4)\
✓ *deliberately* exaggeratedly commented

## References
[1] [Attention Is All You Need, 2017, Vaswani et al.](https://arxiv.org/abs/1706.03762)\
[2] [The Annotated Transformer, 2018, Rush](https://www.aclweb.org/anthology/W18-2509/)\
[3] [http://jalammar.github.io/illustrated-transformer/](http://jalammar.github.io/illustrated-transformer/)
