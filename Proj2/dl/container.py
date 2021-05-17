from .module import Module

class Sequential(Module):
    """ The Sequential Class

        - accepts a list of modules constituting the layers of the net
        - the forward pass iterates through all stored layers and 
        call applies successively all layers; stores the input and the output (implemented in Module)
        - the param function returns a list with all parameters of 
        the neural net
        - the add function allows the addition of a layer after the 
        model was defined by Sequential
        - __getitem__ returns the module with index i
        - summary prints the names of all layers
        - the backward pass is implemented in Module. 

    """

    def __init__(self, *input):
        super().__init__()
        self.module_list=list(input)
        
        
    def forward(self, input):
        output = input
        for i, module in enumerate(self.module_list):
            output = module(output)

        return output

    def param(self):
        params = []
        for module in self.module_list:
            params += module.param()
        return params

    def add(self, module):
        self.module_list.append(module)

    def __getitem__(self, i):
        return self.module_list[i]

    def summary(self):
        print(f"Layer\tN. params")
        for module in self.module_list:
            print(f"{module.__class__.__name__}\t{sum([param.shape[-2] * param.shape[-1] for param in module.param()])}")    
