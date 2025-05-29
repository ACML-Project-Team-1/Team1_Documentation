# Model - CNN

## Describe the model

A Convolutional neural network was implemented as it does a good job at working with image data classification.

### Define the initial model we used

We implemented a CNN model using Pytorch for image classification with 15 output classes.
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, classes=15):
        super(CNN, self).__init__() 
```
The model includes three convolutional layers, first one takes in an RGB image then applies 16 filters of size 5x5. Then the rest of the layers specialise in increasing the depth while using ReLu and Max Pooling.

We used ReLu as the activation function in all the intermediate layers because it introduces non-linearity to help the netowrk to learn more complex atterns, It is also computationally efficient and it helps address the vanishing gradient problem often encountered with other activations like Sigmoid.


### Train the model

The amount of epochs the model needs to be trained for is defined. The training data is split into its label and data. Forward propagation is done, the loss is calculated and backpropagation is done. Afterwards, weights are tuned. The model gets saved into a reusable state.

## Python Libraries 

Pytorch was used in addition to torch vision to define layers and preprocess and transform the data.
