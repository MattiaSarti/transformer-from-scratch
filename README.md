# transformer-from-scratch
## How I Reimplemented the Original Transformer [1] From Scratch
This was a self-exercise to learn about NLP: a PyTorch implementation that narrowly followed an already existing one [2] - with some small differences, though - to prove myself I could build a Transformer from scratch and train it. After gaining theoretical intuition behind the architecture logic [3], I went through the documentation of each line of code [2] reproducing such steps while adundantly documenting - and sometimes customizing - them.\
I was mostly interested in the architecture, and that's my favourite part in the repository.\
Following [2], I trained it on the IWSLT 2016 TED talk translation task to see it work:\
Ours\
Google Translate (as of 6 p.m. on December 21st)

### Code Style:
✓ pylint-compliant\
✓ flake8-compliant\
✓ (deliberately) exaggeratedly commented

### References:
[1] Attention Is All You Need, 2017, Vaswani et al.\
[2] The Annotated Transformer, 2018, Rush\
[3] http://jalammar.github.io/illustrated-transformer/