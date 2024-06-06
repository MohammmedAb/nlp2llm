# Attention Mechanism

## Resources
- [CS224N Lec 7](https://www.youtube.com/watch?v=wzfWHP6SXxY)
- [CS224N Lec 8](https://youtu.be/gKD7jPAdbpE?si=sGuw0MPBtwWnvMH0)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)
- [](https://github.com/bentrevett/pytorch-seq2seq/blob/main/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb)

## Notes:
### The problem with Seq2Seq:
- **The bottleneck problem:** The encoder compresses all the information of the input sequence into a finale hidden state.

### Attention Mechanism:
- **The Core Idea:** on each step of the secoder, we allow the decoder to look at the entire input sequence, then compare the hidden state of the decoder with each hidden state of the encoder using **dot product** to get the **attention scores**. Based on the attention scores, we calculate probability distribution (**attention distribution**) using **softmax**, and then use the attention distribution to calculate the **Attention output**. The attention output is wheighted average of the encoder hidden states, based on the attention scores. The attention output mostly contains information from the hidden states that received high attention.
![image](https://github.com/MohammmedAb/MLOps-Pipeline-Bike-Trip-Duration/assets/83492447/83ccb257-8bf5-4d83-9155-6345e845a7b1)

#### Equations:
- **$h_i$** is the hidden state of the encoder
- **$s_t$** is the hidden state of the decoder at time step t
- **$e^t = [s_t^T h_1, s_t^T h_2, ..., s_t^T h_n]$** is the attention scores at time step t
    - for $s_t$, how much attention it payes to each hidden state of the encoder
- **$\alpha^t = softmax(e^t)$** is the attention distribution at time step t
- **$a_t = \sum_{i=1}^{n} \alpha_i^t h_i$** is the attention output at time step t after the weighted sum of the encoder hidden states
- **$[a_t, s_t]$** is the input to the decoder at time step t, after concatenating the attention output $a_t$ with the hidden state of the decoder $s_t$

#### Attention Variants:
- **Basic Dot Product Attention:** $e^t = [s_t^T h_1, s_t^T h_2, ..., s_t^T h_n]$
- **Multiplicative Attention:** $e^t = [s_t^T W h_1, s_t^T W h_2, ..., s_t^T W h_n]$
    - We put an extra weight matrix W to learn the relationship between the hidden states of the encoder and the decoder (Parts of the hidden states that are important)
- **Reduced rank multiplicative attention:** $e_i = s^T (U^T V)h_i$ 
    - Where U and V are low rank skinny matrices
    - We reduce the rank of the weight matrix W to reduce the number of parameters
- **Additive Attention:** $e_i = v^T tanh(W_1 h_i + W_2 s)$ 
    - We use a feed forward neural network to learn the relationship between the hidden states of the encoder and the decoder
    - $W$ Learnanble weight matrix
    - $v$ Learnable vector that acts as a query or attention weight
    - **The dot product $v^T tanh(W_1 h_i + W_2 s)$ computes the similarity or alignment between the query vector $v$ and the transformed concatenated term. it measure how well teh encoder hidden state $h_i$ and the decoder hidden state $s$ align or match.**

#### Attention is a general DL concept:
> Given a set of vector **values**, and a vector **query**, attention is a technique to compute a weighted sum of the values, dependent on the query. 
You can think of attention as a kind of memory access mechanism, that the weights sum that the attention calculates gives you kind of **selective summary** of information contained in the values, where the query determines which values to focus on.
