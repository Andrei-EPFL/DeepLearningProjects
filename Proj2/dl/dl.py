#!/usr/bin/env python
import torch
from torch.nn.parameter import Parameter
torch.set_grad_enabled ( False )

class Module(object):
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def param(self):
        return []

class ReLU(Module):

    def forward(self, input):
        return torch.clamp(input, min=0)

    def gradient(self, input):
        """ Return the derivative of the activation """

        output = torch.zeros_like(input)
        output[output > 0] = -1

        return output

    def __call__(self, input):
        return self.forward(input)
    
class Tanh(Module):

    def forward(self, input):
        return torch.tanh(input, min=0)

    def gradient(self, input):
        """ Return the derivative of the activation """
        
        return 1. / torch.pow(torch.cosh(input), 2)
    
    def __call__(self, input):
        return self.forward(input)
        

class Sigmoid(Module):

    def forward(self, input):
        return torch.sigmoid(input)

    def gradient(self, input):
        """ Return the derivative of the activation """
        s = torch.sigmoid(input)
        return  s * (1 - s)

    def __call__(self, input):
        return self.forward(input)


class Linear(Module):

    def __init__(self, input_shape, output_units, batch_size):

        #Initialize weights

        self.weights = Parameter(torch.empty(size=(batch_size, output_units, input_shape)))
        self.bias = Parameter(torch.empty(size=(batch_size, 1, output_units)))

    def forward(self, input):
        Z = torch.einsum('bkn,bnj->bkj', self.weights, input) + self.bias
        return Z

    def backward(self, *gradwrtoutput):

        raise NotImplementedError

    def param(self):
        return [self.weights, self.bias]


class Sequential(Module):

    def __init__(self, *input):

        self.module_list=input
        self.fwd_save = {}
        for module in self.module_list:
            self.fwd_save[module.__class__.__name__]=[]
        self.x_list = []
        self.params = []
    def forward(self, input):
        output=0
        self.fwd_save['Linear'].append(input)
        for i, module in enumerate(self.module_list):
            self.fwd_save[module.__class__.__name__].append(module.forward(self.fwd_save[i]))
        return self.fwd_save[-1]

    def backward(self, *gradwrtoutput):

        raise NotImplementedError

    def param(self):
        for module in self.module_list:
            self.params += module.param()
        return self.params


if __name__=='__main__':
    m=Sigmoid()
    print(m.__class__.__name__)