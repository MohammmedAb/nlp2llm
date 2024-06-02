## Resources 
- [a history of seq2seq learning](https://github.com/christianversloot/machine-learning-articles/blob/main/from-vanilla-rnns-to-transformers-a-history-of-seq2seq-learning.md)
- [Wikipedia: RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network)
- [rnn-effectiveness](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Recurrent Neural Networks and Language Models](https://www.youtube.com/watch?v=y0FqGWbfkQw&list=PLw3N0OFSAYSEC_XokEcX8uzJmEZSoNGuS&index=14)
- [Pytorch tutorial on RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial)

## Plan:
1. Youtube Video about RNN: Recurrent Neural Networks and Language Models
2. Karpathy's blog post about RNN effectiveness
3. Pytorch tutorial on RNN
    - Do more and more mini projects with RNN
    - At least 3 projects
        - [*] Classifying Names with a Character-Level RNN
        - [ ] [Generating Names with a Character-Level RNN](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)

### To learn:
- [ ] Seq2Seq learning


### Projects:
- [PoS tagging](https://github.com/bentrevett/pytorch-pos-tagging/blob/master/1_bilstm.ipynb)

# Notes:
# RNN:
- RNN is less an architecture and more a concept.
- RNN typically use one hidden layer.

$$h_t = f(Ux_t + Wh_{t-1})$$
- $h_t$ is the hidden state at time t
- $x_t$ is the input at time t
- $U$ is the weight matrix for the input
- $W$ is the weight matrix for the hidden state
- $f$ is the activation function

### Backpropagation through time (BPTT):
- The overall loss is the sum of the loss at each time step.
- But if we are using only the final output from the RNN, then we can use only the final loss to calculate the gradients.
- We need to update three weight matrices: $U$, $W$, and $V$.
- We need 

### Problems with Regular Neural Networks:
- there's now connection between the words. each word is independent of the other words. (no memory)


### How RNN solves the problem:
- We have a recurrent connection that allow to the hidden layer to recive the current input and the previous hidden state.

## Language Models:
> A language model is a probability distribution over sequences of words.

$$P(w_1, w_2,..., w_n) = \prod_{i=1}^{n} P(w_i | w_1, w_2, ..., w_{i-1})$$

**Used in:**
- Speech recognition
- Machine translation
- Auto completion
- Text generation

### Training RNN Language Model:
1. Start with a corpus and segment it into chunks or sentences or paragraphs, etc.
2. We convert the words into word embeddings
3. We make a vocabulary of all the words in the corpus.
4. We train the RNN to predict the next word in the sequence using a softmax activation function to get the probability of each word being the next word.
### Text Generation:
**Autoregressive Generation:**
We begin with feeding the RNN an initial token (like `<start>`) and then we sample from the probability distribution to get the next word.

### Loss Function:
We use the **Cross Entropy** Loss function to calculate the loss.
- it's the negative log likelihood of the correct word.
- We calculate the loss at each time step and sum them up and average them to get the total loss.

### Softmax Activation Function:
- It converts the output of the RNN into a probability distribution.

### Large Model Evaluation:
Popular metrics for evaluating language models is called **Perplexity**. It is the **cross entropy loss exponentiated**.


## Long-Range Dependencies:
> Long-range dependencies are the dependencies between words that are far apart in the sequence.
This is a problem for Vanilla RNNs because they have a hard time learning long-range dependencies.

### Issues with Vanilla RNNs:
- as the sequence gets longer, information from the beginning of the sequence gets diluted.
- **Vanishing Gradient Problem**: The gradients become very small and the model stops learning.

## Long Short-Term Memory (LSTM):
> LSTM is a type of RNN architecture that is designed to learn long-range dependencies.

## Gating Mechanism:
Gating is a key feature in modern neural networks including LSTMs, GRUs and sparselygated deep neural networks
-  It dynamically controls the flowof the past information to the current state at each time instant. 
-  it is also well known that these
gates prevent the vanishing (and exploding) gradient
problem inherent to traditional RNNs