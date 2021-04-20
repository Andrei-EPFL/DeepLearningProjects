from torch import empty, set_grad_enabled
set_grad_enabled(False)

class nTensor(empty(0).__class__):
    def __init__(self, *args, created_by=None, **kwargs):
        super().__init__()
        self.created_by = created_by

    def set_created_by(self, instance):
        self.created_by = instance
