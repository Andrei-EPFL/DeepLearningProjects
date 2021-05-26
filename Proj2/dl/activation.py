from torch import empty, set_grad_enabled
set_grad_enabled(False)

from .module import Module
from .ntensor import nTensor

def sigmoid(x):
    return 1. / (1 + (-x).exp())

class _Activation(Module):
    """
    _Activation class

    Description:
    
        - inherits from the Module class;
    

    Attributes:
    
    grad: nTensor
        The gradient of the output of the
        activation with respect the input tensor.


    Methods:
    
    backward:
        Computes the gradient with respect the input and stores it

    """
    def __init__(self):
        super().__init__()
        self.grad = 0

    def backward(self, gradwrtoutput):
        """
        Computes the gradient with respect the input and 
        stores it (implemented in Module and nTensor)


        Parameters:

        gradwrtoutput: nTensor
            gradient with respect the output


        Returns: nTensor
        
        the gradient with respect the input
        """
        return nTensor(tensor=gradwrtoutput.tensor * self.grad.tensor)

class ReLU(_Activation):
    """ 
    The ReLU activation function
    
    Description:
    - inherits from _Activation


    Definition:
    - 0, for x <= 0
    - slope * x, for x > 0
    
    
    Attributes:
    
    slope: double

    
    Methods:
    
    forward:
        Computes and stores the gradient; computes the output
    """
    def __init__(self, slope=1):
        super().__init__()
        self.slope = slope

    def forward(self, input):
        """
        Computes and stores the gradient; computes the output
        
        Description:
        - the gradient is stored in self.grad


        Parameters:

        input: nTensor
            Input to the activation


        Returns: nTensor

        the value of the function
        """
    
        self.grad = nTensor(tensor=empty(size=self.input.tensor.shape).fill_(0))
        self.grad.tensor[self.input.tensor > 0] = self.slope
        return nTensor(tensor=self.slope * self.input.tensor.clamp(min=0), created_by=self)
    
class Tanh(_Activation):
    """
    The hyperbolic tangent

    Description:
    - inherits from _Activation


    Methods:
    
    forward:
        Computes and stores the gradient; computes the output
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        """
        Computes and stores the gradient; computes the output
        
        Description:
        - the gradient is stored in self.grad


        Parameters:

        input: nTensor
            Input to the activation


        Returns: nTensor

        the value of the function
        """
    
        self.grad = nTensor(tensor=1. / self.input.tensor.cosh().pow(2))
        return nTensor(tensor=self.input.tensor.tanh(), created_by=self)
    
class Sigmoid(_Activation):
    """ 
    The sigmoid activation

    Description:
    - inherits from _Activation


    Methods:
    
    forward:
        Computes and stores the gradient; computes the output
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        """
        Computes and stores the gradient; computes the output
        
        Description:
        - the gradient is stored in self.grad


        Parameters:

        input: nTensor
            Input to the activation


        Returns: nTensor

        the value of the function
        """
    
        s = sigmoid(self.input.tensor)
        self.grad = nTensor(tensor=s * (1 - s))
        return nTensor(tensor=s, created_by=self)
