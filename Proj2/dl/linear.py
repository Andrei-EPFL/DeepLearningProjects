import math
from torch import empty, set_grad_enabled
set_grad_enabled(False)

from .module import Module
from .ntensor import nTensor

class Linear(Module):

    def __init__(self, input_features, output_features):

        super().__init__()

        #Initialize weights
        self.sqrtk = math.sqrt(1 / input_features)
        self.weights = nTensor(tensor=empty(size=(output_features, input_features)).uniform_(-self.sqrtk, self.sqrtk))
        self.bias = nTensor(tensor=empty(size=(output_features,)).uniform_(-self.sqrtk, self.sqrtk))
        self.zero_grad()

    def forward(self, input):
        self.input = input
        s = input.tensor.matmul(self.weights.tensor.t()).squeeze() + self.bias.tensor
        self.output = nTensor(tensor=s, created_by=self)
        return self.output

    def backward(self, gradwrtoutput):
        grad_s = gradwrtoutput
        grad_x = nTensor(tensor=grad_s.tensor.matmul(self.weights.tensor))

        self.weights.grad = self.weights.grad + (grad_s.tensor[:, :, None] * self.input.tensor[:, None, :]).mean(axis=0)
        self.bias.grad = self.bias.grad + grad_s.tensor.mean(axis=0)
    
        return grad_x

    def param(self):
        return [self.weights, self.bias]
