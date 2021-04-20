#!/usr/bin/env python
from torch import empty, set_grad_enabled, manual_seed
import math
set_grad_enabled(False)

def sigmoid(x):
    return 1. / (1 + (-x).exp())

class nTensor(empty(0).__class__):
    def __init__(self, *args, created_by=None, **kwargs):
        super().__init__()
        self.created_by = None

    def set_created_by(self, instance):
        self.created_by = instance
        
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
            if param.grad is None:
                param.grad = nTensor(size=param.shape).fill_(0)
            else:
                param.grad.fill_(0)


class LossMSE(Module):

    def __init__(self):
        super(Module, self).__init__()
        self.error = None

    def forward(self, prediction, target):
        self.input = prediction
        self.error = (prediction - target)
        self.output =  self.error.pow(2).mean()
        self.output.set_created_by(self)
        return self.output
    
    def backward(self):
        #return -2 * self.error # this gives - twice the gradient as pytorch, weird
        
        grad = self.error
        module = self.input.created_by
        while module:
            grad = module.backward(grad)
            module = module.input.created_by
        return self.error
    
class ReLU(Module):

    def __init__(self, slope=1):
        super(Module, self).__init__()
        self.slope = slope

    def forward(self, input):
        self.input = input
        self.grad = nTensor(size=input.shape).fill_(0)
        self.grad[input > 0] = self.slope
        self.output = self.slope * input.clamp(min=0)
        self.output.set_created_by(self)
        return self.output

    def backward(self, *gradwrtoutput):
        grad, = gradwrtoutput
        return grad * self.grad
    
class Tanh(Module):
    def __init__(self):
        super(Module, self).__init__()

    def forward(self, input):
        self.input = input
        self.grad = 1. / input.cosh().pow(2)
        self.output = input.tanh()
        self.output.set_created_by(self)
        return self.output
    
    def backward(self, *gradwrtoutput):
        grad, = gradwrtoutput
        return grad * self.grad     

class Sigmoid(Module):
    def __init__(self):
        super(Module, self).__init__()

    def forward(self, input):
        self.input = input
        s = sigmoid(input)
        self.grad = s * (1 - s)
        self.output = s
        self.output.set_created_by(self)
        return self.output

    def backward(self, *gradwrtoutput):
        grad, = gradwrtoutput
        return  grad * self.grad

class Linear(Module):

    def __init__(self, input_features, output_features):

        super(Module, self).__init__()

        #Initialize weights
        self.sqrtk = math.sqrt(1 / input_features)
        self.weights = nTensor(size=(output_features, input_features)).uniform_(-self.sqrtk, self.sqrtk)
        self.bias = nTensor(size=(output_features,)).uniform_(-self.sqrtk, self.sqrtk)
        
        self.zero_grad()

    def forward(self, input):
        self.input = input
        s = input.matmul(self.weights.t()).squeeze() + self.bias
        self.output = s
        self.output.set_created_by(self)
        return self.output

    def backward(self, *gradwrtoutput):
        grad_s, = gradwrtoutput
        grad_x = grad_s.matmul(self.weights)

        self.weights.grad = self.weights.grad + (grad_s[:, :, None] * self.input[:, None, :]).mean(axis=0)
        self.bias.grad = self.bias.grad + grad_s.mean(axis=0)
    
        return grad_x

    def param(self):
        return [self.weights, self.bias]

class Sequential(Module):

    def __init__(self, *input):
        super(Module, self).__init__()
        self.module_list=list(input)
        
        
    def forward(self, input):
        self.input = input
        self.input.set_created_by(None)
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
