""" This file contains the implemenation of the Loss Class """

from .module import Module
from .ntensor import nTensor

class LossMSE(Module):
    """ Creates a criterion that measures the Mean Squared Error (MSE).
        L_n (x_n, y_n) = ( x_n - y_n ) ^ 2, for one sample n

        For multiple samples:
        L = ( 1 / Ns ) * \Sigma_{n = 0} ^ {n = Ns} (L_n) 

        In addition to the Module's fields, it stores the error.

        The forward function:
            - takes as input two nTensors: prediction and target,
        i.e. the output of the Neural Net and the true label corresponding
        to the data samples, respectively. The order of the two has to be
        kept when the function is called, because the nTensor prediction
        is stored in the self.input.
            - the output is a nTensor containing the value of the loss;
        it is stored in self.output

        The backward function: 
            - can take anything as input, but it will not be used;
        the reason why there is an input was to simplify the implementation
        of the backward function from the nTensor class.
            - the output is a nTensor representing the error defined as 
                error_n (x_n, y_n) = ( x_n - y_n )
    """
    def __init__(self):
        super().__init__()
        self.error = None

    def forward(self, prediction, target):
        self.error = nTensor(tensor=(prediction.tensor - target.tensor))
        return nTensor(tensor=self.error.tensor.pow(2).mean(), created_by=self)

    def backward(self, gradwrtoutput=None):
        #return -2 * self.error # this gives - twice the gradient as pytorch, weird
        return self.error
