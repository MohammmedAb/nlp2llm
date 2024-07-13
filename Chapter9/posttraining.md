# Aligning Large Language Models

## Things to Learn
- [x] Parameter Efficient Training
    - [x] LoRA
    - [x] QLoRA
- [x] Direct Preference Optimization (DPO)
- [x] RLHF

## Resources
- [Aligning LLM with Human: A Survey](https://github.com/GaryYufei/AlignLLMHumanSurvey)
- [PEFT Hugging Face](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)

### Parameter Efficient Training
Problems with regular fine-tuning:
- We must train the full network, wich is computationally expensive.

#### LoRA
![](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5dfbd169-eb7e-41e1-a050-556ccd6fb679_1600x672.png)
- **LoRA** (Low Rank Adaptation) is a method that reduces the number of parameters that need to be fine-tuned.
> To make fine-tuning more efficient, LoRA’s approach is to represent the weight updates with two smaller matrices (called update matrices) through low-rank decomposition. These new matrices can be trained to adapt to the new data while keeping the overall number of changes low. The original weight matrix remains frozen and doesn’t receive any further adjustments. To produce the final results, both the original and the adapted weights are combined.


- We don't touch the original model's parameters, but we add a low-rank matrix that is fine-tuned. that will alloy us to easily switch between two different fine-tuning models just by changing the parameters of A and B metrices **instead of reloading the W matrix again.**
- LoRA can be applied to any subset of weight matrices in a neural network to reduce the number of trainable parameters
- for simplicity and further parameter efficiency, in Transformer models LoRA is typically applied to attention blocks only
- The resulting number of trainable parameters in a LoRA model depends on the size of the low-rank update matrices, which is determined mainly by the rank r and the shape of the original weight matrix.
- You can load the base model and the LoRA model be separately, but this may encounter some latency issues during inference. 
- You can also merge the two models into one after training, which will reduce the latency during inference. but this will prevent you from switching between the two adaptation models easily.

#### QLoRA
> QLoRA (Quantized Low Rank Adaptation) is a further extension of LoRA that **quantizes** the low-rank matrices to reduce the memory footprint and computational cost of the adaptation process.  

- Typically, parameters of trained models are stored in a 32-bit format, but QLoRA compresses them to a 4-bit format

**Quantization:** Technique that is helpful in reducing the size of the model by converting high precision data to low precision. In simple terms, it converts datatype of high bits to fewer bits.

### Direct Preference Optimization (DPO)
> Direct Preference Optimization (DPO) is a method that directly optimizes the model’s using human preferences without going through reinforcement learning.

- , DPO recasts the alignment formulation as a simple loss function that can be optimised directly on a dataset of preferences ${(x, y_w, w_l)}$
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/pref_tuning/data.png)

Shortcomings of DPO:
- It tends to quickly overfit on the preference dataset
- Creating these datasets is a time consuming and costly endeavour.

How to solve these problems?
- **Identity Preference Optimisation (IPO):** Variant of DPO which adds a regularisation term to the DPO loss and enables one to train models to convergence without requiring tricks like early stopping.
- **Kahneman-Tversky Optimisation (KTO):**  Defines the loss function entirely in terms of individual examples that have been labelled as "good" or "bad" 