import math

from .module import Module
from .ntensor import nTensor

class Linear(Module):

    def __init__(self, input_features, output_features):

        super(Linear, self).__init__()

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
