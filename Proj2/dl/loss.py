from .module import Module
from .ntensor import nTensor

class LossMSE(Module):

    def __init__(self):
        super().__init__()
        self.error = None

    def forward(self, prediction, target):
        self.input = prediction
        self.error = nTensor(tensor=(prediction.tensor - target.tensor))
        self.output = nTensor(tensor=self.error.tensor.pow(2).mean(), created_by=self)
        return self.output
    
    def backward(self, gradwrtoutput=None):
        #return -2 * self.error # this gives - twice the gradient as pytorch, weird
        return self.error
