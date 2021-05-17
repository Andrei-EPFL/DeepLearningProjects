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
        - forward function stores the input the tensor (implemented in Module),
        the output tensor (implemented in Module) and returns the output tensor;
        it calls the "function" method specific to each Activation
        - backward function returns the gradient with respect
        the input and stores it (implemented in Module)
    """
    def __init__(self):
        super().__init__()
        self.grad = 0

    def forward(self, input):
        return self.function()
    
    def backward(self, gradwrtoutput):
        return nTensor(tensor=gradwrtoutput.tensor * self.grad.tensor)

    def function(self):
        raise NotImplementedError

class ReLU(_Activation):
    """ 
        The ReLU activation function:
        - 0, for x < 0
        - slope * x, for x > 0
        the slope can be given at the initialization;

        - in method "function" the gradient is computed 
        and stored; finally it returns the value of the function
    """
    def __init__(self, slope=1):
        super().__init__()
        self.slope = slope

    def function(self):
        self.grad = nTensor(tensor=empty(size=self.input.tensor.shape).fill_(0))
        self.grad.tensor[self.input.tensor > 0] = self.slope
        return nTensor(tensor=self.slope * self.input.tensor.clamp(min=0), created_by=self)
    
class Tanh(_Activation):
    """
        The hyperbolic tangent:

        - in method "function" the gradient is computed 
        and stored; finally it returns the value of the function
    """
    def __init__(self):
        super().__init__()

    def function(self):
        self.grad = nTensor(tensor=1. / self.input.tensor.cosh().pow(2))
        return nTensor(tensor=self.input.tensor.tanh(), created_by=self)
    
class Sigmoid(_Activation):
    """ 
        The sigmoid activation:

        - in method "function" the gradient is computed 
        and stored; finally it returns the value of the function
    """
    def __init__(self):
        super().__init__()

    def function(self):
        s = sigmoid(self.input.tensor)
        self.grad = nTensor(tensor=s * (1 - s))
        return nTensor(tensor=s, created_by=self)
