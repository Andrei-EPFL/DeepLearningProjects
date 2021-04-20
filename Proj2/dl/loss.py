from .module import Module

class LossMSE(Module):

    def __init__(self):
        super(LossMSE, self).__init__()
        self.error = None

    def forward(self, prediction, target):
        self.input = prediction
        self.error = (prediction - target)
        self.output =  self.error.pow(2).mean()
        self.output.set_created_by(self)
        return self.output
    
    def backward(self):
        #return -2 * self.error # this gives - twice the gradient as pytorch, weird
        
        grad = self.error
        module = self.input.created_by
        while module:
            grad = module.backward(grad)
            module = module.input.created_by
        return self.error
