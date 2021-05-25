from .ntensor import nTensor

from torch import empty, set_grad_enabled
set_grad_enabled(False)

class Module(object):
    def __init__(self):
        self.input = None
        self.output = None

    def __call__(self, *args, **kwargs):
        self.input = args[0]
        self.output = self.forward(*args, **kwargs)
        return self.output

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, gradwrtoutput):
        grad = gradwrtoutput
        module = self.output.created_by

        while module:
            grad = module.backward(grad)
            module.input.grad = grad.tensor
            module = module.input.created_by
        return grad

    def param(self):
        return []

    def zero_grad(self):
        for param in self.param():
            if param.grad is None:
                param.grad = empty(size=param.tensor.shape).fill_(0)
            else:
                param.grad.fill_(0)
