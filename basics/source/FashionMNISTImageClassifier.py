from torch import nn


""" A classifier model for Fashion MNIST Image 
We creat this module by sub classing nn.Module, a Module can contain other modules
The common use modules:
- nn.Flatten: The nn.Flatten layer is commonly used to convert multi-dimension input tensor to a one-dimension tensor.
               As a result, it's often used as input layer. In this example, the flatten layer transform a 2D 28x28 
               image into a contiguous array of 784 pixel values.
- nn.Linear: The linear layer is a module that applies a linear transformation on the input by using the stored 
              weights and biases of the layer.               
- nn.ReLU: This module use a Non-linear activations which can create the complex mappings between the model's inputs 
              and outputs. They are applied after linear transformations to introduce non-linearity, helping neural 
              networks learn a wide variety of phenomena.
- nn.Sequential: nn.Sequential is an ordered container of modules. The data is passed through all the modules in the 
               same order as defined. You can use sequential containers to put together a network of modules.
- nn.Softmax: This module is used to normalize raw values in [-infty, infty] to values in [0, 1]. As a result, this 
              module is often used as output layer for multiple class classification. The outputs of this layer 
              represents the model's predicted probabilities for each class. The 'dim' parameter indicates the 
              dimension along which the values must sum to 1. It means the output possibility values for each class 
              are dependent of this dimension, and the sum is 1. For example:
              input values: -0.5, 1.2, -0.1, 2.4
              SoftMax output values: 0.04, 0.21, 0.05, 0.70
              The sum is 1
- nn.Sigmoid: This module also normalize raw values in [-infty, infty] to values in [0, 1] or [-1,1]. This module is 
              often used as output layer for binary class classification. The output possibility values for each class 
              are independent, and the sum can be any value. For example: 
              input value: -0.5, 1.2, -0.1, 2.4
              Sigmoid output values: 0.37, 0.77, 0.48, 0.91
              The sum is 2.53

In this model, we use nn.ReLU between our linear layers, but there's other activations to introduce non-linearity in 
your model.
"""
class FashionMNISTImageClassifier(nn.Module):
    def __init__(self):
        super(FashionMNISTImageClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
