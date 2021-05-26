""" This file contains the implemenation of the Loss Class """

from .module import Module
from .ntensor import nTensor

class LossMSE(Module):
    """
    Module that creates a criterion that measures the Mean Squared Error (MSE).

    Description:

        - inherits from the Module base class
        
        - error_n (x_n, y_n) = ( x_n - y_n )

        - L_n (x_n, y_n) = ( x_n - y_n ) ^ 2, for one sample n

        - L = ( 1 / Ns ) * \Sigma_{n = 0} ^ {n = Ns} (L_n), for Ns samples
        
        (y_n is the true label and x_n is the prediction)

    
    Attributes:

    error: nTensor
        Stores the error (prediction - target)


    Methods:

    forward:
            Computes the loss given the prediction and the true label

    backward: 
            Returns the gradient with respect the prediction 
    """
    def __init__(self):
        super().__init__()
        self.error = None

    def forward(self, prediction, target):
        """
        Computes the loss given the prediction and the true label
        
        Description:
        
        - The order of the two input nTensors has to be kept when the function is called, because 
        the prediction is stored in the self.input (implemented in Module).
        
        
        Parameters:

        prediction: nTensor
            The output of the Neural Network

        target: nTensor
            The true label corresponding to the data samples 
            

        Returns:   
            
        loss: nTensor
            Containins the value of the loss
            It is stored in self.output (implemented in Module)
        """
        self.error = nTensor(tensor=(prediction.tensor - target.tensor))
        return nTensor(tensor=self.error.tensor.pow(2).mean(), created_by=self)

    def backward(self, gradwrtoutput=None):
        """
        Returns the gradient with respect the prediction 

        Observation:

        - 2 * self.error gives twice the gradient as pytorch
            
        
        Parameters: not used!

        - the reason why there is an input was to simplify the implementation
        of the backward function from the nTensor class.
    

        Returns:
        
        error: nTensor
            Represents the error defined as 
            error_n (x_n, y_n) = ( x_n - y_n )
        """
        return self.error
