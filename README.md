# How I Reimplemented the Original Transformer [1] from Scratch
An exercise to learn about NLP: a PyTorch implementation that followed an already existing one [2] - with some small differences, though - to prove myself I could build a Transformer from scratch and train it.\
After gaining theoretical intuition behind the architecture logic [3], I went through every line of code [2] and the respective libraries' docs, reproducing such steps while customizing, testing and abundantly documenting them.

### Insights:
I was mostly interested in the architecture, but following [2] I also trained it on the German-to-English translation task of the IWSLT 2016 dataset (original, I know...) to see it in action:\
Ours\
Google Translate (as of 6 p.m. on December 21st)

#### Why is it not a good task to apply a Transformer?
mmmm
#### Why did I choose such task anyway, then?
Because I was into the architecture, I just trained it to see it work correctly without employing an external dataset.

### Tests:
✓ Unit Tests: passed\
✓ Integration Tests: passed\
✓ Code Coverage: ...%

### Code Style:
✓ Pylint-compliant\
✓ Flake8-compliant\
✓ (deliberately) exaggeratedly commented

### References:
[1] Attention Is All You Need, 2017, Vaswani et al.\
[2] The Annotated Transformer, 2018, Rush\
[3] http:<span>//jalammar.github.io</span>/illustrated-transformer/

NB: no positional embeddings added to first Decoder tokens!!