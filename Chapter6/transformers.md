# Self-Attention | Transformers 

## Resources
- [ ] [CS224N Leacture 9](https://youtu.be/ptuGllU5SQQ?si=T6p8hBwC88o9IJyd)
- [ ] Attention is All You Need Paper
- [ ] [Karpathy GPT from scratch](https://youtu.be/kCc8FmEb1nY?si=7uIevkmpFFykpEPP)

- [ ] [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

## Notes

### Problems with RNNs
#### Linear interaction distance:
- It hard to learn long-distance dependencies in RNNs, because of the gradient problems.
- Linear order isn't always the best way to think about sentences.

#### Lack of Parallelization:
- Forward and backward passes have `O(sequece_length)` unparallelizable operations.
- Future RNN hidden states can't be computed in full before past RNN hidden states are computed.

## Attention-Based NLP Models

### Self-Attention
IMPORTANT: Attention operates on **Queries $q_T$**, **Keys $k_T$**, and **Values $v_T$**. 

- The (dot product version) slef-attention operation is as follows:
$$e_{ij} = q_i^T k_j$$
    - We transposed the query vector $q_i$ to make the dot product possible: The dot product requires one vector to be a row vector and the other to be a column vector
- The attention scores are then normalized using the softmax function 

#### What's the difference between self-attention and full connected layers?
![](https://pbs.twimg.com/media/Fp6DofVXsAERpA8?format=jpg&name=small)
- In fully connected layers, weights are static with respect to the input
- In self-attention: We have dynamic connecteivity, weights are dynamic between the key and the query, which are dependent on the actual input.

#### The problem if we use the self-attention alone as a bulding block
**The self-attention mechanism is permutation invariant:** which means the mechanism itself treats all positions in the sequence uniformly when computing attention scores. This means that when computing the attention score between any two positions in the sequence, the self-attention mechanism does not inherently prioritize certain positions over others based on their order in the sequence

### Solving the permutation invariance problem: Positional Encoding
> We add a positional encoding to the **input embeddings** to give the model some information about the relative or absolute position of the tokens in the sequence.
- We represent each sequence index as a vector:
$$p_i \in \mathbb{R}^d, \text{ for } i \in \{1, 2, \ldots, T\} $$
- We add the $P_i$ to our inputs. $\tilde{v}_i \tilde{k}_i \tilde{q}_i$ be our old values, keys, and queries.
$ v_i = \tilde{v}_i + p_i$
$k_i = \tilde{k}_i + p_i$
$q_i = \tilde{q}_i + p_i$

#### Position Encoding Vector: What does it look like?
> We let all $p_i$ be learned parameters
- We make a matrix $P \in \mathbb{R}^{d \times T}$, where $T$ is the sequence length and $d$ is the dimension of the model (the embedding dimension).
- Wvery value in the matrix is learnable parameter.

### Adding non-linearity in self-attention
> Note that there are no elements-wise non-linearities in self-attention; staking more self-attention layers will just re-average the value vectors.
- We add a feed-forward network (FNN) after the self-attention layer to add non-linearity.
$$ m_i = \text{MLP}(output_i)$$
$$ = W_2 * \text{ReLU}(W_1 \times output_i + b_1) + b_2$$
- Where $output_i$ is the output of the self-attention layer.
- $W_1, W_2,$ are the weights matrices 
- $b_1, b_2$ are the bias vectors

### Ensure the model don't look at the future: (Masking)
> We use masking to prevent the model from looking at the future tokens.
- we maske the future in the decoder
- We mask out attention scores to future words by setting the attention scores to $-\infty$
$$ e_{ij} = \begin{cases} q_i^T k_j, & j < i \\ -\infty, & j \geq i  \end{cases} $$

if the key index is less than the query index, we keep the attention score, otherwise, we set it to $-\infty$. this insures that the model doesn't look at the future tokens.

## Transformer Model
There are more elements we need to make a transformer model:
- **key-query-value attention:** How do we get the $k, q, v$ vectors from a single word embedding?
- **Multi-head attention**
- **Triks to help with training:** Layer normalization, residual connections, Scaling the dot product.

### Key-Query-Value Attention
> Key and query Matrices turn the input embeddings into one that is more suitable for calculating similarity.
Values matrix are the embeddings that we will return as the output of the attention mechanism.

#### How key-query-value attention is computed?
I don't understand this part yet. 

### Multi-Head Attention