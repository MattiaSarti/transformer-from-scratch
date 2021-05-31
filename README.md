# How I Reimplemented the Original Transformer from Scratch

An exercise to learn about Transformers: a PyTorch implementation of the original model [[1](https://github.com/MattiaSarti/transformer-from-scratch#references)] that followed an already existing one [[2](https://github.com/MattiaSarti/transformer-from-scratch#references)] - with some small differences, though - to prove myself I could build a Transformer from scratch and train it.\
After gaining theoretical intuition behind the architecture logic [[3](https://github.com/MattiaSarti/transformer-from-scratch#references)], I went through every line of code [[2](https://github.com/MattiaSarti/transformer-from-scratch#references)] and the respective libraries' docs, reproducing such steps while customizing, testing and abundantly documenting them.

## Purpose
I was interested in understanding the details of the [architecture implementation](https://github.com/MattiaSarti/transformer-from-scratch/tree/main/transformer/architecture) - to then propose a modification of the position encoding mechanism, as done in [this other repository](https://github.com/MattiaSarti/rethinking-position-encoding-in-transformers) - but following [[2](https://github.com/MattiaSarti/transformer-from-scratch#references)] I also [trained and evaluated](https://github.com/MattiaSarti/transformer-from-scratch/tree/main/transformer/training_and_inference) the model on a toy source-to-target copy task to ensure its proper functioning. An example of resulting translations follows, reproducible by running ```python try_toy_task.py``` in the root directory of the repository.\
*<...coming soon...>*

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
