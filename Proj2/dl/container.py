from .module import Module

class Sequential(Module):

    def __init__(self, *input):
        super(Sequential, self).__init__()
        self.module_list=list(input)
        
        
    def forward(self, input):
        self.input = input
        self.input.set_created_by(None)
        output = input
        for i, module in enumerate(self.module_list):
            output = module(output)
        self.output = output
        return self.output

    def backward(self, *gradwrtoutput):
        grad, = gradwrtoutput
        for i, module in enumerate(self.module_list[::-1]):
            grad = module.backward(grad)

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
