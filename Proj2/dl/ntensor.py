from torch import empty, set_grad_enabled
set_grad_enabled(False)

class nTensor():
    """ 
        The nTensor class:
        - takes as input when initialized, the module
        that created the stored tensor.
        - it stores:
            - a tensor
            - a gradient (as tensor) with respect the tensor
            - the module that created the tensor
            - the shape of the tensor
        - set_created_by takes as input the instance that created 
        the tensor and it stores in the created_by field.
        - backward pass
    """
    def __init__(self, tensor=None, created_by=None):

        self.tensor = tensor
        self.created_by = created_by
        self.grad = None
        self.shape = self.tensor.shape

    def set_created_by(self, instance):
        self.created_by = instance

    def backward(self):
        module = self.created_by
        grad = nTensor(tensor=empty(size=self.tensor.shape).fill_(1))
        self.grad = grad.tensor
        while module:
            grad = module.backward(grad)
            module.input.grad = grad.tensor
            module = module.input.created_by
        return grad

    def __getitem__(self, id_):
        return self.tensor[id_]

    def __len__(self):
        return len(self.tensor)
