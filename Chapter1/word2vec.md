# Study Plan: 
- [x] Watch the Stanford CS224n Lecture 1
- [x] Watch the Stanford CS224n Lecture 2
- [*] Read the original word2vec paper
- [*] Implement a word2vec model

# Papers to read:
- [Efficient Estimation of Word Representations in
Vector Space (original word2vec paper)](https://arxiv.org/pdf/1301.3781.pdf)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
# hands-on
- [ ] Implement a word2vec model from scratch
    - [ ] Implement the Skip-gram model
    - [ ] Implement the Negative Sampling technique

# Resources
- [Stanford CS224n Lecture 1](https://youtu.be/rmVRLeJRkl4?si=j6rvlRmsrJvP1zo4)
- [Stanford CS224n Lecture 2](https://youtu.be/ERibwqs9p38)
- [Word2vec PyTorch implementation - Notebook](https://github.com/enesozeren/nlp-playground/blob/main/word_embeddings/word2vec_training.ipynb)
- [Word2vec PyTorch implementation - blog post](https://medium.com/@enozeren/word2vec-from-scratch-with-python-1bba88d9f221)


# Notes
# Lecture 1 - Intro 
## Traditional NLP
The Problem with traditional NLP:
- Missing new meanings of words and it's impossible to keep up-to-date
- Requires human labor to create and adapt
- Using one hot encoding to represent words is not efficient
- We need a huge number of vectors to represent all the words in the dictionary

## Word Embeddings
> Dense vectors of real numbers, one per word in your vocabulary, where similar words have similar vectors.

## Word2Vec Algorithm (2013)
### **Idea:**
- We have a large **corpus ("body")** of text
- Every word in a fixed vocabulary is represented by a vector
- Go through each **position t** in the text, which has a **center word c** and a **context ("outside") word o**
- Use the similarity if the word vector for c and o to calculate the probability of o given c (or vice versa)
- Keep adjusting the word vectors to maximize this probability
### What is the objective function? 
> For each position t = 1, 2, ..., T in the text, we want to predict context words within a window of fixed size m, given center word $w_j$  

**Data likelihood:** $L(\theta) = \prod_{t=1}^{T} \prod_{-m \leq j \leq m, j \neq 0} P(w_{t+j} | w_t; \theta)$

The likelihood is calculated as taking the product of using each word as a center word and then the product of each word and a window around that of the probability of predicting that context word given the center word.

**Objective function:** 
$J(\theta) = -\frac{1}{T} \log L(\theta) $ 

$J(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \sum_{-m \leq j \leq m, j \neq 0} \log P(w_{t+j} | w_t; \theta)$  

- $J(\theta)$ is the objective function that we want to minimize
- $\theta$ represents all the variables to be optimized, in this case, the word embeddings
- $T$ is the total number of words in the corpus 
- The negative sign and the log are applied to the likelihood L(θ) for mathematical convenience and numerical stability
- The log likelihood is expanded as a double summation over each center word w_t (summing from t=1 to T) and each context word $w_{t+j}$ within a window of size m around the center word.
- The inner summation is over the range -m to +m (inclusive), skipping $j=0$ to exclude the center word itself.
- $P(w_{t+j} | w_t; θ)$ represents the probability of observing the context word $w_{t+j}$ given the center word $w_t$ and the current model parameters θ.

The objective function (Sometimes called the loss function) is the negative log likelihood of the data averaged over the whole corpus.  

The goal is to minimize this objective function in order to maximize the predictive accuracy of the model.

By minimizing this average negative log likelihood, the model learns to assign high probabilities to the actual context words that appear around each center word, thereby maximizing its predictive accuracy. The optimal parameters θ that minimize J(θ) give us the final learned word embeddings.

### How to calculate the probability of a context word given a center word $P(w_{t+j} | w_t; \theta)$?
We will use two vectors for each word:
1. **v** for the center word $v_w$
2. **u** for the context word $u_w$  

Then the probability of a context word given a center word is calculated as:
$P(o|c) = \frac{exp(u_{o}^T v_{c})}{\sum_{w\in V} exp(u_{w}^T v_{c})}$

> What we using here is called the **softmax function**. It takes a vector and converts them into things between 0 and 1 that sum up to 1. This way is called (naive softmax) because it's very expensive to compute.

Where:
- $P(o|c)$ is the probability of observing a context word o given a center word c.
- $u_o$ is the output vector representation of the context word o.
- $v_c$ is the input vector representation of the center word c.
- $u_o^T v_c$ is the **dot product between the output vector of the context word and the input vector of the center word, which measures the compatibility or similarity between the two words.**
- $exp(u_o^T v_c)$ is the exponential function applied to the dot product, which converts the compatibility score into a positive value **(to avoid negative probabilities).**
- $Σ_{w∈V} exp(u_w^T v_c)$ is the sum of the exponential of the dot products between the input vector v_c and the output vectors u_w of all words w in the vocabulary V. **This sum serves as a normalization term to ensure that the probabilities sum up to 1.**

### How to optimize the objective function (train the model)?
What we want to do is to minimize the objective function $J(\theta)$ **by adjusting the model parameters θ**, which in this case are the word vectors

We use **Gradient Descent** to minimize the objective function. 

# Lecture 2 - Neural Classifiers

### Optimization: Gradient Descent
The idea of **Gradient Descent**: from the current position ($\theta$), we compute the gradient of the objective function with respect to the model parameters ($J(\theta)$) and then update the **parameters in the opposite direction of the gradient to minimize the objective function**.

Gradient Descent Equation:
$\theta = \theta - \alpha \nabla_{\theta} J(\theta)$

Where:
- $\theta$ is the model parameters (word vectors in this case)
- $\alpha$ is the learning rate (step size)
- $\nabla_{\theta} J(\theta)$ is the gradient of the objective function with respect to the model parameters

> This is just the basic idea of Gradient Descent. nobody uses this in practice because it's too slow. There are many variations of Gradient Descent that are more efficient and faster, such as **Stochastic Gradient Descent (SGD)**, **Mini-batch Gradient Descent**, and **Adam**.

### The problems with the basic Gradient Descent algorithm:
- $J(\theta)$ is a function of all windows in the corpus, which can be very large
- $\nabla_{\theta} J(\theta)$ is very expensive to compute
- Optimization can be slow and inefficient

### Stochastic Gradient Descent (SGD)

We can use **Stochastic Gradient Descent (SGD)** to minimize the objective function. The idea is to estimate the gradient of the objective function **using a small subset of the data (a single window in this case) and update the parameters based on this estimate.**

## Word2Vec Algorithm family
There are two main algorithms in the Word2Vec family:
1. **Skip-grams (SG)**
Predict context words (outside words) given the center word
2. **Continuous Bag of Words (CBOW)**
Predict the center word from the context words

## Negative Sampling - More Efficient Training

There are two ways to train the Word2Vec model:
1. **Naive Softmax** (as we discussed earlier)
calculates a probability distribution over the entire vocabulary for the target word, given the context. The softmax function is used to normalize the output scores, converting them into probabilities. 
- **Problem:** It's computationally expensive to compute the softmax function for all words in the vocabulary, especially for large vocabularies.
2. **Negative Sampling**
Instead of predicting the probability distribution over the entire vocabulary, negative sampling focuses on distinguishing the target word from a small set of randomly sampled "negative" words.

### How does Negative Sampling work?
> Main idea: Train a binary logistic regression classifier that distinguishes the target word from a set of randomly sampled negative words.

$$J_{t}(\theta) = log \sigma(u_{o}^T v_{c}) + \sum_{k=1}^{K} \mathbb{E}_{j \sim P_n(w)}[log \sigma(-u_{j}^T v_{c})]$$

### Dimensionality Reduction
> The word vectors learned by the Word2Vec model are typically high-dimensional (e.g., 100-300 dimensions). We can use **dimensionality reduction techniques** to reduce the dimensionality of the word vectors while preserving their semantic relationships.

**Single Value Decomposition (SVD)**: A common technique for dimensionality reduction is to use Singular Value Decomposition (SVD) to factorize the word vectors into a lower-dimensional space.

The idea: Given a matrix of word vectors $$X$$, we can factorize this matrix into three matrices: $$X = U \Sigma V^T$$

## GloVe: Global Vectors for Word Representation (2014) 
GloVe is another popular word embedding model that learns word vectors by factorizing the word co-occurrence matrix.
