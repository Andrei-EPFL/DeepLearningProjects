from .ntensor import nTensor

from torch import empty, set_grad_enabled
set_grad_enabled(False)

class Module(object):
    """
    Base class

    Description:
        - Most classes inherit from Module
        - This class serves as a template for the child classes.


    Attributes:

    input: nTensor
        Stores the input of a Module
        (i.e. the input of the forward method)
    
    output: nTensor
        Stores the output of a Module
        (i.e. the output of the forward method)

    
    Methods:

    __call__:
        Allows the application of an instance on a nTensor
        Stores the input and the output of the forward method
        Returns the output of the forward method

    forward:
        Not implemented

    backward:
        Triggers the backward pass of the Neural Network

    param:
        Returns an empty list

    zero_grad:
        Sets the gradients with respect all parameters of Neural Network to zero.
    """

    def __init__(self):
        self.input = None
        self.output = None

    def __call__(self, *args, **kwargs):
        """
        Description:
            - Allows the application of an instance on a nTensor
            - Stores the input and the output of the forward method
        
        
        Parameters:
            - same as for forward method

        args: list of arguments
            The first one should be the input

        kwargs: list of keyword arguments


        Returns:
           
        output: nTensor 
            The output of the forward method

        """
        self.input = args[0]
        self.output = self.forward(*args, **kwargs)
        return self.output

    def forward(self, *args, **kwargs):
        """
        Must be implemented for each child class specifically
        """

        raise NotImplementedError

    def backward(self, gradwrtoutput):
        """
        Triggers the backward pass of the Neural Network

        Description:
            - used in this form only by Sequential and any other
            class that defines a Neural network
            - stores in the grad field of each input nTensor of all the layers
            the gradient with respect that input

        Parameters:

        gradwrtoutput: nTensor
            The gradient with respect to the output (prediciton) of the Neural Network
            More specifically, the gradient of the loss with respect the prediction

        Returns:

        grad: nTensor
            The gradient with respect the input nTensor of the Neural Network.
            More specifically the gradient of the loss with respect the input data set.
        """

        grad = gradwrtoutput
        module = self.output.created_by

        while module:
            grad = module.backward(grad)
            module.input.grad = grad.tensor
            module = module.input.created_by
        return grad

    

    #def param(self):
    
    #    return []
    
    def param(self):
        """
        Finds the parameters of the module, returns empty list if
        none are found.
        
        Can be overwritten by the child classes when necessary.
        """
        params = []
        for key, module in self.__dict__.items():
            try:
                params += module.param()
            except:
                continue
        return params

    def zero_grad(self):
        """
        Sets the gradients with respect all parameters of Neural Network to zero.
        If necessary it initializes the gradients to zero.
        """
        for param in self.param():
            if param.grad is None:
                param.grad = empty(size=param.tensor.shape).fill_(0)
            else:
                param.grad.fill_(0)
