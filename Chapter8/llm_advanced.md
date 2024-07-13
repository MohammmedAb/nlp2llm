# Advanced Topics in Large Language Models


## Concepts to Learn
- [x] Scaling Laws

- [x] [Traning Hypermeters](https://rentry.org/llm-training#training-hyperparameters)

- [x] Learn About the Different Architectures of LLMs
    - llama
    - Mistral MOE

## Resources
- [Chinchilla's wild implications](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications)
- [Learn about computation and memory usage for transformers](https://blog.eleuther.ai/transformer-math/)
- [UMass CS685 Scalling laws class](https://www.youtube.com/watch?v=N7n66FL7wqM)
- [Llama from scratch](https://github.com/jzhang38/TinyLlama)
- [Mistral MOE](https://huggingface.co/blog/moe)

## Project Idea:
- [ ] Name, Place, Animal & Thing Game

# Notes 

# Training Hyperparameters
- **Batch size**: The number of samples that will be processed in one iteration before the model's weights are updated. 
- **Number of Epochs**: The number of times the model will see the entire dataset.

## Gradient Accumulation
> Gradient accumulation is a mechanism to split the original batch size into smaller mini-batches. We preform multiple stpes of computation without updating the model's weights. instead, we keep track of the gradients (accumulating them) and update the model's weights after a the mini-batches are processed.

# Scaling Laws
> Scaling laws are a set of mathematical equations that govern the the dependence of overfitting on model/datset size and the dependence of traning speed on model size. These relationships allow us to determine the optimal allocation of a fixed amount of compute resources.
## OpenAI Scaling Laws 2020
### Model Performance Dependence on What?
Model preformance depends most strongly on scasle, which consists of three factors:
- **Number of model parameters $N$**
- **Size of the dataset $D$**
- **Amount of compute $C$**
---
1. First, for optimally compute-efficient training, most of the increase should go towards **increased model size**
2. Second, **batch size**. that larger batch sizes are beneficial as compute increases, but not as dramatically as model size.
3. Third, Serial Steps. which is the number of training iterations, and it is the least important factor.

### Overfitting
Performance improves predictably as long as we scale up N and D in tandem, **but enters a regime of diminishing returns if either N or D is held fixed while the other increases**.
To avoid the performance penalty of overfitting, **we must increase them both on this ratio: $N^{0.74} / D$** 

#### Sample Efficiency
Large models are more sample-efficient than small models, reaching the same level of performance with fewer optimization steps (Figure 2) and using fewer data points

### Summary of Scaling Laws
- For models with a limited number of parameters, trained to convergence on sufficiently large datasets:
$$ L(N) = (N_c/N)^{\alpha N}$$
- For large models trained with a limited dataset with early stopping:
$$ L(D) = (D_c/D)^{\alpha D}$$

## DeepMind Scaling Laws 2022 (Chinchilla Paper)
$$ L = \text{Language Model test loss (Cross-Entropy loss)} $$
$$ D = \text{Dataset size (number of traning tokens)} $$
$$ N = \text{\# model parameters} $$
$$ C = \text{Compute (FLOPS)} $$ 
### Takeaways
- **You should increase model and data size at the same rate**
- **The paper suggests an optimal ratio of about 20 tokens of training data per model parameter**


