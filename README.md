# transformer-from-scratch
## How I Reimplemented the Original Transformer [1] from Scratch
An exercise to learn about NLP: a PyTorch implementation that followed an already existing one [2] - with some small differences, though - to prove myself I could build a Transformer from scratch and train it. After gaining theoretical intuition behind the architecture logic [3], I went through every line of code [2] and the respective docs, reproducing such steps while adundantly documenting - and sometimes customizing - them.\
I was mostly interested in the architecture, and that's my favourite part in the repository, but following [2] I also trained it on the IWSLT 2016 TED talk dataset representing an English-to-German translation task to see it work:\
Ours\
Google Translate (as of 6 p.m. on December 21st)

### Code Style:
✓ Pylint-compliant\
✓ Flake8-compliant\
✓ (deliberately) exaggeratedly commented

### References:
[1] Attention Is All You Need, 2017, Vaswani et al.\
[2] The Annotated Transformer, 2018, Rush\
[3] http:<span>//jalammar.github.io</span>/illustrated-transformer/
