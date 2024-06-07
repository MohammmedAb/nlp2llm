# Transformers | Self-Attention

## Resources
- [] [CS224N Leacture 9](https://youtu.be/ptuGllU5SQQ?si=T6p8hBwC88o9IJyd)
- [] [Karpathy GPT from scratch](https://youtu.be/kCc8FmEb1nY?si=7uIevkmpFFykpEPP)

- [] [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

## Notes

### Problems with RNNs
#### Linear interaction distance:
- It hard to learn long-distance dependencies in RNNs, because of the gradient problems.
- Linear order isn't always the best way to think about sentences.

#### Lack of Parallelization:
- Forward and backward passes have `O(sequece_length)` unparallelizable operations.
- Future RNN hidden states can't be computed in full before past RNN hidden states are computed.