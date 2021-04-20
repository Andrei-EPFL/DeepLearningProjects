from .module import Module
from .ntensor import nTensor

def sigmoid(x):
    return 1. / (1 + (-x).exp())

class _Activation(Module):
    def __init__(self):
        super(_Activation, self).__init__()
        self.grad = 0

    def forward(self, input):
        self.input = input
        self.output = self.function(input)
        self.output.set_created_by(self)
        return self.output
    
    def backward(self, gradwrtoutput):
        return gradwrtoutput * self.grad

    def function(self, input):
        raise NotImplementedError

class ReLU(Module):

    def __init__(self, slope=1):
        super(ReLU, self).__init__()
        self.slope = slope
        self.grad = 0

    def forward(self, input):
        self.input = input
        self.grad = nTensor(size=input.shape).fill_(0)
        self.grad[input > 0] = self.slope
        self.output = self.slope * input.clamp(min=0)
        self.output.set_created_by(self)
        return self.output

    def backward(self, gradwrtoutput):
        return gradwrtoutput * self.grad
    
class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()
        self.grad = 0

    def forward(self, input):
        self.input = input
        self.grad = 1. / input.cosh().pow(2)
        self.output = input.tanh()
        self.output.set_created_by(self)
        return self.output
    
    def backward(self, gradwrtoutput):
        return gradwrtoutput * self.grad

class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.grad = 0

    def forward(self, input):
        self.input = input
        s = sigmoid(input)
        self.grad = s * (1 - s)
        self.output = s
        self.output.set_created_by(self)
        return self.output

    def backward(self, gradwrtoutput):
        return gradwrtoutput * self.grad
