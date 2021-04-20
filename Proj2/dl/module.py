from .ntensor import nTensor

class Module(object):
    def __init__(self):
        self.input = None
        self.output = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def param(self):
        return []

    def zero_grad(self):
        for param in self.param():
            if param.grad is None:
                param.grad = nTensor(size=param.shape).fill_(0)
            else:
                param.grad.fill_(0)
