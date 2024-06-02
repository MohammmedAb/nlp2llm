# Overfitting
neural network that overfits is a neural network that has learned the training data too well. This means that the neural network has learned the noise in the training data instead of making decisions based only in the true signal. 

# Regularization
Regularization is a subfield of methods for getting the model generalize (prevent overfitting).

## Early Stopping
Early stopping is a method that stops the training of the model when the validation loss starts to increase. This is a way to prevent the model from overfitting.

How do we know when to stop? the only real way know is to run the model on data that isn't in the traning set. In some circumstances, if you used the test set for knowing when to stop, you could overfit to the test set. As a general rule, you don’t use it to control training. You use a validation set instead.

- We use early stopping to make the neural network to train only in the signal and ignore the noise.
- Early stopping is the cheapest form of regularization.

## Dropout
Dropout is a regularization technique that prevents overfitting by randomly setting some neurons to zero during the training. it makes the a big neural network to act like a little one, by capturing only the big, obvious, high-level features. 

- this regularization technique is generally accepted as the state-of-the-art method for preventing overfitting for the vast majority of neural networks.
- Dropout is only applied during training. During inference (evaluation), all neurons are used, and the outputs are not scaled
- This is handled automatically by PyTorch. When you call `model.eval()`, the dropout layers are deactivated.

# Types of Gradient Descent
## Stochastic Gradient Descent (SGD)
The simplest form of gradient descent. It uses single training examples at a time to update the model weights. 

- Advantages:
    - Faster Iterations
    - Memory Efficiency

- Disadvantages: 
    - Slow Convergence
    - Noisy Updates

## Mini-batch Gradient Descent
It uses a small batch of training examples to update the model weights.

- Advantages:
    - Provides a good trade-off between the noisy updates of SGD and the smooth updates of Batch Gradient Descent.
    - faster computation than SGD.
    - more stable and faster convergence compared to SGD.

- Disadvantages:
    - Requires more memory than SGD to store the mini-batches.

## Batch Gradient Descent
It uses the entire training set to update the model weights.

- Advantages:
    - Stable Convergence.
    - Efficient Use of Computational Resources: Takes advantage of highly optimized matrix operations on the entire dataset.
    
- Disadvantages:
    - Requires significant memory to process the entire dataset at once.
    - Can be very slow for large datasets.

# Activation Functions
An activation function is a function that is applied to the neurons in a layer during the forward pass. It introduces non-linearity to the model, allowing it to learn complex patterns in the data.

## Standard hidden-layer activation functions
- Sigmoid: 
    - Range: (0, 1)
    - lets you interpret the output of any individual neuron as a probability. Thus, people use this nonlinearity both in hidden layers and output layers.

- Tanh:
    - Range: (-1, 1)
    - this aspect of negative correlation is powerful for hidden layers; on many problems, tanh will outperform sigmoid in hidden layers.

- ReLU:
    - Range: (0, ∞)
    - ReLU is the most popular activation function for deep neural networks.

## Standard output-layer activation functions
- No activation function:
    - used for regression problems where the output is a continuous value.
    - Where the range of the output is something other than probability.

- Sigmod:
    - Range: (0, 1)
    - used for binary classification problems.

- Softmax:
    - Range: (0, 1)
    - used for multi-class classification problems (Probebility distribution callculations).
    