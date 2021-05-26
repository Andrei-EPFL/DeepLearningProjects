""" This file contains the implemenation of the Linear Class """

import math
from torch import empty, set_grad_enabled
set_grad_enabled(False)
from .module import Module
from .ntensor import nTensor

class Linear(Module):
    """
    Module that applies a linear transformation 
    
    Description:

        - OUT = IN * W^T  + B, where * is in general a matrix multiplication
        - inherits from the Module base class


    Attributes:

    sqrtk: double
        The square root of (1 / input_features)
        Used to sample the the values of the weights and the bias

    weights (W): nTensor, shape D_{L} x D_{L-1}
        The weights (W) of the linear transformation

    bias (B): nTensor, shape D_{L}
        The bias (B) of the linear transformation
        
        (where L is the current layer and L-1 is the previous layer)
   
    biasbool: bool
        If true the bias is also used
        If false only the weights are used 
        

    Methods:

    forward:
        Applies the linear transformation

    backward:
        Computes the gradient with respect the input
    
    param:
        Returns the list with the weights and the bias

    """
    def __init__(self, input_features, output_features, biasbool=True):
        """
        Parameters:
        
        input_features: int
            Number of input units

        output_features: int
            Number of output units

        biasbool: bool
            If true the bias is also used
            If false only the weights are used        


        Description:

        - the bias and the weights are initially sampled from a uniform distribution
        with the variance inversely proportional to the number of input features, in order
        to diminish the effect of vanishing gradient in the backward pass. 
        - the gradients with respect the weights and the biases
        are set to zero using self.zero_grad() (implemented in the base Module).
        """
        
        super().__init__()

        #Initialize weights
        self.sqrtk = math.sqrt(1 / input_features)
        self.weights = nTensor(tensor=empty(size=(output_features, input_features)).uniform_(-self.sqrtk, self.sqrtk))
        self.biasbool = biasbool
        if self.biasbool:
            self.bias = nTensor(tensor=empty(size=(output_features,)).uniform_(-self.sqrtk, self.sqrtk))
        self.zero_grad()

    def forward(self, input):
        """
        Applies the linear transformation
        OUT = IN * W^T  + B

        Parameters:

        input: nTensor (shape N x D_{L-1}, N - the number of samples in the batch) 
            Input to the Linear class

        
        Returns:
        
        output: nTensor (of shape N x D_{L})
            The result of applying the linear transformation on the input
        """

        
        s = input.tensor.matmul(self.weights.tensor.t()).squeeze()
        if self.biasbool:
            s += self.bias.tensor

        return nTensor(tensor=s, created_by=self)

    def backward(self, gradwrtoutput):
        """
        Computes the gradient with respect to the input

        Description:
        - updates the gradients with respect the weights and biases
        - checks whether there is only one sample in the batch
        and adapt the multiplications


        Parameters:

        input: nTensor
            The gradient with respect to the output of the forward function;

        
        Returns:

        output: nTensor
            The gradient with respect to the input of the forward function
        """

        grad_s = gradwrtoutput
        grad_x = nTensor(tensor=grad_s.tensor.matmul(self.weights.tensor))

        if len(grad_s.shape) == 2 and len(self.input.shape) == 2:
            self.weights.grad = self.weights.grad + (grad_s.tensor[:, :, None] * self.input.tensor[:, None, :]).mean(axis=0)
            if self.biasbool:
                self.bias.grad = self.bias.grad + grad_s.tensor.mean(axis=0)
        elif len(grad_s.shape) == 1 and len(self.input.shape) == 1:
            self.weights.grad = self.weights.grad + (grad_s.tensor[:, None] * self.input.tensor[None, :]).mean(axis=0)
            if self.biasbool:
                self.bias.grad = self.bias.grad + grad_s.tensor
        else:
            raise ValueError(f"The shape of the grad_s is {grad_s.shape} and the shape of the input is {self.input.shape}. Not broadcastable!")

        return grad_x

    def param(self):
        """
        Returns:
        
        list with weights and bias nTensors.
        """
    
        if self.biasbool:
            return [self.weights, self.bias]
        
        return [self.weights]
        