from .module import Module

class Sequential(Module):
    """
    The Sequential Class

    Description:
    - inherits from the Module base class
    - allows the creations of a Neural Network


    Attributes:

    module_list: list of modules
        Constitutes the layers of the Neural Network


    Methods:
    
    forward:
        Iterates through all stored layers and 
        applies them successively.
        
    param:
        Returns a list with all parameters of the Neural Network
    
    add:
        Adds another layer after the model was defined by Sequential
    
    __getitem__:
        Returns the module with index i

    summary:
        Prints information about the Neural Network
    """

    def __init__(self, *input):
        super().__init__()
        self.module_list=list(input)
        
    def forward(self, input):
        """
        Iterates through all stored layers and 
        applies them successively.
        
        Description:
        - stores the input and the output (implemented in Module)


        Parameters:

        input: nTensor
            Input to the model


        Returns: nTensor shape
            Prediction of the Neural Network
        """

        output = input
        for i, module in enumerate(self.module_list):
            output = module(output)

        return output

    def param(self):
        """
        Description:
        - it iterates through all layers and asks for their parameters

        
        Returns:
        
        a list with all parameters of the Neural Network
        """

        params = []
        for module in self.module_list:
            params += module.param()
        return params

    def add(self, module):
        """
        Adds another layer after the model was defined by Sequential
        """

        self.module_list.append(module)

    def __getitem__(self, i):
        """
        Parameters:

        i: int
            index of the required module
        

        Returns:

        the module with index i
        """
        return self.module_list[i]

    def summary(self):
        """
        Prints the names of all layers and the 
        number of corresponding parameters
        """

        print(f"Layer\tN. params")
        for module in self.module_list:
            print(f"{module.__class__.__name__}\t{sum([param.shape[-2] * param.shape[-1] for param in module.param()])}")    
