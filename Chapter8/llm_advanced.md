# Advanced Topics in Large Language Models


## Concepts to Learn
- [ ] Scaling Laws

- [x] [Traning Hypermeters](https://rentry.org/llm-training#training-hyperparameters)

- [ ] Learn About the Different Architectures of LLMs
    - llama
    - Mistral MOE

## Resources
- [Chinchilla's wild implications](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications)
- [Learn about computation and memory usage for transformers](https://blog.eleuther.ai/transformer-math/)
- [Llama from scratch](https://github.com/jzhang38/TinyLlama)
- [Mistral MOE](https://huggingface.co/blog/moe)

# Notes 

# Training Hyperparameters
- **Batch size**: The number of samples that will be processed in one iteration before the model's weights are updated. 
- **Number of Epochs**: The number of times the model will see the entire dataset.

## Gradient Accumulation
> Gradient accumulation is a mechanism to split the original batch size into smaller mini-batches. We preform multiple stpes of computation without updating the model's weights. instead, we keep track of the gradients (accumulating them) and update the model's weights after a the mini-batches are processed.
