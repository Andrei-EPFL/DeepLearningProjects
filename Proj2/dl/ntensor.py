from torch import empty, set_grad_enabled
set_grad_enabled(False)

class nTensor():
    def __init__(self, tensor=None, created_by=None):

        self.tensor = tensor
        self.created_by = created_by
        self.grad = None

    def set_created_by(self, instance):
        self.created_by = instance

    def backward(self):
        module = self.created_by
        grad = None

        while module:
            grad = module.backward(grad)
            module = module.input.created_by
        
        return grad