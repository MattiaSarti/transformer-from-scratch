# How I Reimplemented the Original Transformer [1] from Scratch

An exercise to learn about Transformers: a PyTorch implementation of the original model that followed an already existing one [2] - with some small differences, though - to prove myself I could build a Transformer from scratch and train it.\
After gaining theoretical intuition behind the architecture logic [3], I went through every line of code [2] and the respective libraries' docs, reproducing such steps while customizing, testing and abundantly documenting them.

### Purpose
I was interested in the [architecture definition](https://github.com/MattiaSarti/transformer-from-scratch/tree/main/transformer/architecture), but following [2] I also [trained](https://github.com/MattiaSarti/transformer-from-scratch/tree/main/transformer/training_and_inference) the model on a toy source-to-target copy task to ensure its proper functioning.\
An example of resulting translations follows, reproducible by running ```python try_toy_task.py``` from the root directory of the repository.\
<... CMD screenshot or stdout logs here ...>

### Tests
✓ Unit Tests: passed\
✓ Integration Tests: passed\
✓ Code Coverage: ...%

### Code Style
✓ Pylint-compliant (2.5.3)\
✓ Flake8-compliant (3.8.4)\
✓ *deliberately* exaggeratedly commented

### References
[1] [Attention Is All You Need, 2017, Vaswani et al.](https://arxiv.org/abs/1706.03762)\
[2] [The Annotated Transformer, 2018, Rush](https://www.aclweb.org/anthology/W18-2509/)\
[3] [http://jalammar.github.io/illustrated-transformer/](http://jalammar.github.io/illustrated-transformer/)
