from torch import empty, set_grad_enabled
set_grad_enabled(False)

from .module import Module
from .ntensor import nTensor

def sigmoid(x):
    return 1. / (1 + (-x).exp())

class _Activation(Module):
    """ _Activation class
        - inherits from the Module class;
        - self.grad (nTensor) contains the gradient of the output 
        of the activation as function of the input;
        - backward function returns the gradient with respect
        the input and stores it (implemented in Module)
    """
    def __init__(self):
        super().__init__()
        self.grad = 0

    def backward(self, gradwrtoutput):
        return nTensor(tensor=gradwrtoutput.tensor * self.grad.tensor)

class ReLU(_Activation):
    """ 
        The ReLU activation function:
        - 0, for x < 0
        - slope * x, for x > 0
        the slope can be given at the initialization;

        - in method "forward" the gradient is computed 
        and stored; finally it returns the value of the function
    """
    def __init__(self, slope=1):
        super().__init__()
        self.slope = slope

    def forward(self, input):
        self.grad = nTensor(tensor=empty(size=self.input.tensor.shape).fill_(0))
        self.grad.tensor[self.input.tensor > 0] = self.slope
        return nTensor(tensor=self.slope * self.input.tensor.clamp(min=0), created_by=self)
    
class Tanh(_Activation):
    """
        The hyperbolic tangent:

        - in method "forward" the gradient is computed 
        and stored; finally it returns the value of the function
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.grad = nTensor(tensor=1. / self.input.tensor.cosh().pow(2))
        return nTensor(tensor=self.input.tensor.tanh(), created_by=self)
    
class Sigmoid(_Activation):
    """ 
        The sigmoid activation:

        - in method "forward" the gradient is computed 
        and stored; finally it returns the value of the function
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        s = sigmoid(self.input.tensor)
        self.grad = nTensor(tensor=s * (1 - s))
        return nTensor(tensor=s, created_by=self)
