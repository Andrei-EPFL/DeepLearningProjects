""" This file contains the implemenation of the Linear Class """

import math
from torch import empty, set_grad_enabled
set_grad_enabled(False)

from .module import Module
from .ntensor import nTensor

class Linear(Module):
    """ Creates a Linear layer which applies a linear transformation 
        of the input nTensor.
        
            OUT = IN * W^T  + B, where * is in general a matrix multiplication,
        W is a nTensor of weights and B is a nTensor with bias values.

            The weights (of shape D_{L} x D_{L-1}, where L is the current layer and L-1 is the previous layer)
        and the bias values (of shape D_{L}) are initially sampled from a uniform distribution
        with the variance inversely proportional to the number of input features, in order
        to diminish the effect of vanishing gradient in the backward pass. 
        
            At the initialization, the gradients with respect the weights and the biases
        are set to zero using self.zero_grad(), implemented in the base Module.

        The forward function:
            - takes as input one nTensor (of shape N x D_{L-1}, where N is the number of samples
        in the batch and stored in self.input) on which the linear transformation is applied;

            - the output is a nTensor (of shape N x D_{L}, stored in self.output) is obtained
        after the linear transformation

        The backward function: 
            - takes as input a nTensor which is the gradient with respect
        to the output of the current forward function;

            - the output is a nTensor representing the gradient with respect the input
        nTensor of the forward function
            - updates the gradients with respect the weights and biases
            - stores the gradient with respect the input nTensor in self.input.grad

        The param function:
            - returns the weights and bias nTensors.
    """
    def __init__(self, input_features, output_features):

        super().__init__()

        #Initialize weights
        self.sqrtk = math.sqrt(1 / input_features)
        self.weights = nTensor(tensor=empty(size=(output_features, input_features)).uniform_(-self.sqrtk, self.sqrtk))
        self.bias = nTensor(tensor=empty(size=(output_features,)).uniform_(-self.sqrtk, self.sqrtk))
        self.zero_grad()

    def forward(self, input):
        s = input.tensor.matmul(self.weights.tensor.t()).squeeze() + self.bias.tensor
        return nTensor(tensor=s, created_by=self)

    def backward(self, gradwrtoutput):
        grad_s = gradwrtoutput
        grad_x = nTensor(tensor=grad_s.tensor.matmul(self.weights.tensor))

        if len(grad_s.shape) == 2 and len(self.input.shape) == 2:
            self.weights.grad = self.weights.grad + (grad_s.tensor[:, :, None] * self.input.tensor[:, None, :]).mean(axis=0)
            self.bias.grad = self.bias.grad + grad_s.tensor.mean(axis=0)
        elif len(grad_s.shape) == 1 and len(self.input.shape) == 1:
            self.weights.grad = self.weights.grad + (grad_s.tensor[:, None] * self.input.tensor[None, :]).mean(axis=0)
            self.bias.grad = self.bias.grad + grad_s.tensor
        else:
            raise ValueError(f"The shape of the grad_s is {grad_s.shape} and the shape of the input is {self.input.shape}. Not broadcastable!")

        return grad_x

    def param(self):
        return [self.weights, self.bias]
