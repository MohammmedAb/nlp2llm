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

### Self-Attention
IMPORTANT: Attention operates on **Queries $q_T$**, **Keys $k_T$**, and **Values $v_T$**. 

- The (dot product version) slef-attention operation is as follows:
$$e_{ij} = q_i^T k_j$$
    - We transposed the query vector $q_i$ to make the dot product possible: The dot product requires one vector to be a row vector and the other to be a column vector
- The attention scores are then normalized using the softmax function 

### What's the difference between self-attention and full connected layers?
![](https://pbs.twimg.com/media/Fp6DofVXsAERpA8?format=jpg&name=small)
- In fully connected layers, weights are static with respect to the input
- In self-attention: We have dynamic connecteivity, weights are dynamic between the key and the query, which are dependent on the actual input.