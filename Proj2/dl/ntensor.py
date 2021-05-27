from torch import empty, set_grad_enabled
set_grad_enabled(False)

class nTensor():
    """ 
    The nTensor class:


    Description:
            - takes as input a tensor and the module that created that tensor


    Attributes:

    tensor: tensor
        Stores the PyTorch tensor.
    
    created_by: reference
        Stores a reference to the module that has as output the stored tensor.
        
    grad: tensor
        Stores the gradient with respect the stored tensor.    

    shape: torch.Size
        Stores the shape of the stored tensor


    Methods:
    
    set_created_by:
        Changes the field created_by

    backward:
        Triggers the backward propagation
    
    __getitem__:
        Returns the value from index id_ of the stored tensor

    __len__:
        Returns the length of the stored tensor
    """

    def __init__(self, tensor=None, created_by=None):

        self.tensor = tensor
        self.created_by = created_by
        self.grad = None
        self.shape = self.tensor.shape

    def set_created_by(self, instance):
        """
        Changes the field created_by

        Parameters:
        
        instance: reference
            The reference to the module that has as output the stored tensor
        """

        self.created_by = instance

    def backward(self):
        """
        Triggers the backward propagation

        Description:
            - stores in the grad field of each input nTensor of all the layers
            the gradient with respect that input
            - in PyTorch the initialization with 1 of the grad variable
            is not done automatically.

        Returns:
        
        grad: nTensor
            The gradient with respect the stored tensor
        """

        module = self.created_by
        grad = nTensor(tensor=empty(size=self.tensor.shape).fill_(1))
        self.grad = grad.tensor
        while module:
            grad = module.backward(grad)
            module.input.grad = grad.tensor
            module = module.input.created_by
        return grad

    def __getitem__(self, id_):
        """
        __getitem__:
            Returns the value from index id_ of the stored tensor 
        """

        return self.tensor[id_]

    def __len__(self):
        """
        __len__:
            Returns the length of the stored tensor
        """

        return len(self.tensor)
