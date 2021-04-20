#!/usr/bin/env python
from torch import empty, set_grad_enabled, manual_seed
import math
set_grad_enabled(False)

def sigmoid(x):
    return nTensor(tensor=1. / (1 + (-x()).exp()))


class nTensor(empty(0).__class__):
    def __init__(self, *args, created_by=None, **kwargs):
        super().__init__()
        self.created_by = created_by

    def set_createdby(self, instance):
        self.created_by = instance

    # def backward(self):
    #     if self.created_by == None:
    #         raise RuntimeError("Error message")
    #     else:
    #         if tensor is loss:
    #             module =self.created_by
    #             grad = module.backward()
    #         else:
    #             grad = ones_like(self)
            
    #         while module:
    #             grad = module.backward(grad)
    #             module = module.input.created_by
        
    #     return grad
            
class oldnTensor():
    def __init__ (self, tensor=empty(0), created_by=None):
        self.tensor = tensor
        self.created_by = created_by
    
    def set_createdby(self, instance):
        self.created_by = instance

    def backward(self):
        grad = nTensor(tensor=empty(self.tensor.shape).fill_(1))
        module = self.created_by
        while module:
            grad = module.backward(grad)
            module = module.input.created_by
        return grad()

    def __call__(self):
        return self.tensor 
        
class Module(object):
    def __init__(self):
        self.input = None
        self.output = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def param(self):
        return []

    def zero_grad(self):
        for param in self.param():
            if param().grad is None:
                param().grad = empty(size=param().shape).fill_(0)
            else:
                param().grad.fill_(0)


class LossMSE(Module):

    def __init__(self):
        super(Module, self).__init__()
        self.error = None

    def forward(self, prediction, target):
        self.input = prediction
        self.error = nTensor(tensor=prediction() - target(), created_by=self)
        self.output = nTensor(tensor=self.error().pow(2).mean(), created_by=self)
        return self.output
    
    def backward(self, grad):
        #return -2 * self.error # this gives - twice the gradient as pytorch, weird
        return self.error
    
class ReLU(Module):

    def __init__(self, slope=1):

        super(Module, self).__init__()
        self.slope = slope

    def forward(self, input):
        self.input = input
        self.grad = nTensor(tensor=empty(size=input().shape).fill_(0))
        self.grad.tensor[input() > 0] = self.slope
        
        self.output = nTensor(tensor=self.slope * input().clamp(min=0), created_by=self)
        return self.output

    def backward(self, *gradwrtoutput):
        grad, = gradwrtoutput
        return nTensor(tensor=grad() * self.grad())
    
class Tanh(Module):

    def forward(self, input):
        self.input = input
        self.grad = 1. / input.cosh().pow(2)
        self.output = input.tanh()
        self.output.set_createdby(self)
        return self.output
    
    def backward(self, *gradwrtoutput):
        grad, = gradwrtoutput
        return grad * self.grad     

class Sigmoid(Module):

    def forward(self, input):
        self.input = input
        s = sigmoid(input)
        self.grad = nTensor(tensor=s() * (1 - s()))
        self.output = nTensor(tensor=s(), created_by=self)
        return self.output

    def backward(self, *gradwrtoutput):
        grad, = gradwrtoutput
        return nTensor(tensor=grad() * self.grad())

class Linear(Module):

    def __init__(self, input_features, output_features):

        super(Module, self).__init__()

        #Initialize weights
        self.sqrtk = math.sqrt(1 / input_features)
        self.weights = nTensor(tensor=empty(size=(output_features, input_features)).uniform_(-self.sqrtk, self.sqrtk))
        self.bias = nTensor(tensor=empty(size=(output_features,)).uniform_(-self.sqrtk, self.sqrtk))        
        self.zero_grad()


    def forward(self, input):
        self.input = input
        s = nTensor(tensor=input().matmul(self.weights().t()).squeeze() + self.bias())
        self.output = nTensor(tensor=s(), created_by=self)
        return self.output

    def backward(self, *gradwrtoutput):
        grad_s, = gradwrtoutput
        grad_x = nTensor(tensor=grad_s().matmul(self.weights()))

        self.weights.tensor.grad = self.weights().grad + (grad_s().t().mm(self.input()))
        self.bias.tensor.grad = self.bias().grad + grad_s().mean(axis=0)

        return grad_x

    def param(self):
        return [self.weights, self.bias]

class Sequential(Module):

    def __init__(self, *input):
        super(Module, self).__init__()
        self.module_list=list(input)
        
        
    def forward(self, input):
        self.input = input
        output = input
        for i, module in enumerate(self.module_list):
            output = module(output)
        self.output = output
        return self.output

    def backward(self, *gradwrtoutput):
        grad, = gradwrtoutput
        for i, module in enumerate(self.module_list[::-1]):
            grad = module.backward(grad)

    def param(self):
        params = []
        for module in self.module_list:
            params += module.param()
        return params

    def add(self, module):
        self.module_list.append(module)

    def __getitem__(self, i):
        return self.module_list[i]

    def summary(self):
        print(f"Layer\tN. params")
        for module in self.module_list:
            print(f"{module.__class__.__name__}\t{sum([param.shape[-2] * param.shape[-1] for param in module.param()])}")    
