# Learn and Implement LLaMA Model Architecture

## Resources
- [LLaMA 3 Architecture Blog](https://medium.com/@vi.ai_/exploring-and-building-the-llama-3-architecture-a-deep-dive-into-components-coding-and-43d4097cfbbb)
- [TinyLlama](https://github.com/jzhang38/TinyLlama)
- [llama3 from scratch](https://github.com/naklecha/llama3-from-scratch)
- [LLaMA 3 Repo](https://github.com/meta-llama/llama3/blob/main/llama/model.py)
- [Causal Attention Mechanism](https://snawarhussain.com/educational/llms/Causal-Attention-Mechanism-Pure-and-Simple/)
- [LLaMA vinija's AI notes](https://vinija.ai/models/LLaMA/)

- [LLaMA 3 implimentation/debuging](https://github.com/viai957/LexiGenesis)
- [llama-inference](https://github.com/viai957/llama-inference)

## LLaMA 2 Paper notes
### Pretraning
#### Some Architecture Details
- Pre-normalization: Used to imorove the traning stability, we normalize the input of each transformer sub-layer, instead of normalizing the output. we use the **RMSNorm** instead of the LayerNorm (GPT-2).
- RMSNorm (Zhang and Sennrich, 2019):
- SwiGLU Activation Function: A variant of the Gated Linear Unit (GLU) (Shazeer, 2020): 
- rotary positional embeddings (RoPE) (RoPE, Su et al. 2022):
- Grouped-query-attention (GQA): 2305.13245

#### Efficient Implementation Details
- Efficient implementation of the causal multi-head attention: Used to reduce memory usage and runtime. This is achieved by not storing the attention weights and not computing the key/query scores that are masked due to the causal nature of the language modeling task.
- Reducing Activation Recomputation in Large Transformer ModelsKorthikanti et al. (2022)

**Things I have to learn:**
- [x] Causal multi-head attention
- [x] RMSNorm
- [x] SwiGLU Activation Function
- [x] rotary positional embeddings (RoPE)
- [] Grouped-query-attention
- [] KV Cache

##### RMSNorm
> The vanilla network might have issues with the stability of parameters' gradients, (Internal Covariate Shift) it makes traning the network slower, delayed convergence, as the neuorns are forces to re-adjust drasticaly because of the drastic changes in the output of the previous layer. to reduce these issues, we use normalization techniques. RMSNorm is one of these techniques. 
Key differences between RMSNorm and LayerNorm:
- A well-known explanation of the success of LayerNorm is its re-centering and re-scaling invariance property
- RMSNorm only focuses on re-scaling invariance and regularizes the summed inputs simply according to the root mean square (RMS) statistic
- Traditional LayerNorm involves calculating the mean and variance of activations for each layer, which can be computationally intensive, especially in large models or with high-dimensional data
- RMSNorm simplifies this process by only calculating the root mean square (RMS) of the activations, which involves fewer operations than computing both mean and variance
The formula for RMSNorm is:
$$ \bar{a}_i = \frac{a_i}{\text{RMS}(\mathbf{a})}g_i, \quad \text{where} \quad \text{RMS}(\mathbf{a}) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} a_i^2} $$

##### SwiGLU Activation Function
> The SwiGLU activation function is based on the Swish activation function, which is a smooth and non-monotonic function that has been shown to outperform other commonly used activation functions such as ReLU and sigmoid in certain neural network architectures

- It was introduced to improve the performance of transformer models by providing a more efficient and effective way of gating and activating neurons
- Experimental results have shown that SwiGLU can outperform other activation functions such as ReLU, Swish, and GELU (Gaussian Error Linear Units) on certain image classification and language modeling tasks.

The SwiGLU activation function is defined as:
$$ SwiGLU(x) = x * sigmoid(\beta * x) + (1- sigmoid(\beta * x)) * x$$

##### Rotary Positional Encoding (RoPE)
> RoPE encode the absolute position witha rotation matrix and leverage the positional information into the learning process of pre-trained language models.

RoPE allows the model have valuable properties including:
- Seqence length flexibility
- Decaying inter-token dependencies
- The capability of equipping the linear self-attention with relative positional encoding.

What is a rotation matrix: rotation matrix is a matrix that rotates a vector to another vector by some angle

#### KV Cache
> Without KV Cache, the attention mechanism in transformer models has to compute the key-value pairs for each token at each time step. At each generation step we are recalculating the same previous token attention, when we actually just want to calculate the attention for the new token. Which can be computationally expensive and make the inference process slow. KV Cache is a technique used in transformer models to store the key-value pairs of the attention mechanism in memory, which can be reused across multiple time steps to reduce the computational cost of the attention mechanism.

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*uyuyOW1VBqmF5Gtv225XHQ.gif)

**Disadvantage of KV Cache**: KV Cache can increase the memory usage of the model, as it requires storing the key-value pairs of the attention mechanism in memory

Equation of the memory usage of the KV Cache:
$$ 2* \text{percision} * \text{n\_layers} * \text{h\_model} * \text{seqlen} * \text{batch} $$