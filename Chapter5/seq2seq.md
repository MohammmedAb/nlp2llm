# Seq2Seq

## Resources
- [ ] [Sequence to Sequence Learning with Neural Networks](https://github.com/bentrevett/pytorch-seq2seq)
- [ ] [CS224N Lecture 7](https://www.youtube.com/watch?v=wzfWHP6SXxY&list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ&index=8&t=13s)

## Notes:

### The problem before seq2seq:
DNNs can only be applied to problems whose inputs and targets can be sensibly encoded with vectors of fixed dimensionality. It is a significant limitation, since many problems, especially those involving sequences like text or speech, do not naturally fit into this fixed-size vector representation. Sequences can have variable lengths, and encoding them into fixed-size vectors may lead to loss of information or inefficient representations

### Overview of Seq2Seq:
Seq2Seq is a model that consists of two Models: an encoder and a decoder. The encoder reads the input sequence and outputs a initial hidden state for the decoder. The decoder reads the initial hidden state and outputs the translated sequence. The encoder and decoder are trained jointly to maximize the probability of the correct translation given the input sequence.

#### Other applications of Seq2Seq that are not translation:
- Summarization (input: long text, output: short text)
- Question Answering (input: question, output: answer)
- Code Generation (input: natural language, output: code)
- Image Captioning (input: image, output: text) 

#### Training a Neural Translation Model:
- Encode the source sentence with our encoder LSTM
- Feed the final hidden state of the encoder to the target LSTM (Decoder) as the initial hidden state
- Train word by word,by comparing the predicted word to the actual word in the target sentence
- Calculate the loss which is the sum of the negative log likelihoods of the predicted words
- We backpropagate the loss **through the entire network (encoder and decoder)** and update the weights

### Varous decoding strategies:
- Greedy Decoding: At each time step, we choose the word with the highest probability as the output. This is the simplest decoding strategy.
    - Pros: Simple and fast
    - Cons: has no way to undo decisions, and may not find the best overall sequence

- Exhaustive Search Decoding: We consider all possible output sequences and choose the one with the highest probability. 
    - Pros: Guarantees the best output sequence
    - Cons: Computationally expensive, since it requires evaluating all possible sequences

- **Beam Search Decoding**: At each time step, we keep track of the top k most probable sequences (hypotheses). We then expand each hypothesis by considering all possible next words, and keep the top k most probable sequences. We repeat this process until we reach the end of the sequence.
    - Pros: Guarantees a good output sequence, and is more computationally efficient than exhaustive search
    - Stoping Criteria: Different hypotheses may produce <END> token at different time steps. We place it aside and continue expanding other hypothese the beam search. We stop when we reach timestep T (Where T is some predefined number) or when we have at least n completed hypotheses (Where n predefined cutoff)

**We notice a problem with beam search decoding:** longer hypotheses have lower scores than shorter ones. This is because the probabilities are multiplied at each time step, and longer sequences have more probabilities multiplied together. 
To solve this, we can normalize the score by dividing it by the length of the sequence. This is called **length normalization**. 