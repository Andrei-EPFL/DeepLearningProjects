from torch import empty, set_grad_enabled
set_grad_enabled(False)
import dlc_practical_prologue as prologue

def dloss(v, t):
    return -2*(t-v)

def loss(v, t):
    return (t-v).pow(2).sum()


### I am not sure if this is the best way to define the activation function
### Currently this is my best idea, but let's discuss and see what we can do.

def dactivation_function(x, type_="tanh"):
    if type_ == "tanh":
        return 1 - x.tanh().pow(2)
    if type_ == "relu":
        output = empty(x.shape).zero_()
        output[x > 0] = 1
        return output
    print("ERROR: You chose a wrong activation function! Please choose between \"tanh\" and \"relu\"")
    exit()
    return 0

def activation_function(x, type_="tanh"):
    if type_ == "tanh":
        return x.tanh()
    if type_ == "relu":
        return x.relu()
    print("ERROR: You chose a wrong activation function! Please choose between \"tanh\" and \"relu\"")
    exit()
    return 0

class Linear():
    def __init__(self, in_features, out_features):
        self.weights = empty(size=(out_features, in_features)).normal_(mean=0, std=1e-6)
        self.bias = empty(size=(out_features,)).normal_(mean=0, std=1e-6)
        self.sum_var = empty((0,))
        self.x_var = empty((0,))
        self.dl_dw = empty(size=(out_features, in_features)).zero_()
        self.dl_db = empty(size=(out_features,)).zero_()
        self.dl_dw_tot = empty(size=(out_features, in_features)).zero_()
        self.dl_db_tot = empty(size=(out_features,)).zero_()
        self.dl_ds = empty((0,))

    def compute_sum(self, input_x):
        self.sum_var = input_x.mm(self.weights.t()) + self.bias

    def apply_activation_function(self, type_=1):
        self.x_var = activation_function(self.sum_var, type_=type_)

    def compute_dl_ds(self, dl_dx, dsigma_s):
        self.dl_ds = dl_dx.mul(dsigma_s)

    def compute_dl_dw(self, input_x):
        self.dl_dw = self.dl_ds.t().mm(input_x)
        self.dl_dw_tot += self.dl_dw

    def compute_dl_db(self):
        self.dl_db = self.dl_ds.sum(0)
        self.dl_db_tot += self.dl_db

    def set_zero_grad(self):
        self.dl_dw_tot.zero_()
        self.dl_db_tot.zero_()

class Net():
    def __init__(self, nb_layers, nb_features_layer, activation_types, init_nb_features):
        self.nb_layers = nb_layers
        self.nb_features_layer = nb_features_layer
        self.activation_types = activation_types
        self.layers = []

        self.init_nb_features = init_nb_features

        for L in range(self.nb_layers):
            if L == 0:
                self.layers.append(Linear(self.init_nb_features, self.nb_features_layer[L]))
            else:
                self.layers.append(Linear(self.nb_features_layer[L - 1], self.nb_features_layer[L]))

    def forward_pass(self, x0):
        '''
            Forward Pass
            x0 - the input data set having batch_size number of samples
                and D features
        '''

        # nb_layers - 1 hidden layers starting from 0 to nb_layers - 2;
        # the last layer is the prediction
        for L in range(self.nb_layers):
            if L == 0:
                self.layers[L].compute_sum(x0)
            else:
                self.layers[L].compute_sum(self.layers[L - 1].x_var)

            self.layers[L].apply_activation_function(type_=self.activation_types[L])

    def backward_pass(self, x0, target):
        '''
            Backward Pass
            x0 - the input data set having batch_size number of samples
                and D features
            target - the target data set
        '''

        # nb_layers - 1 hidden layers starting from 0 to nb_layers - 2;
        # the last layer is the prediction
        for L in range(self.nb_layers - 1, -1, -1):
            if L == self.nb_layers - 1:
                dl_dx = dloss(self.layers[L].x_var, target)
            else:
                dl_dx = self.layers[L + 1].dl_ds.mm(self.layers[L + 1].weights)

            self.layers[L].compute_dl_ds(dl_dx, dactivation_function(self.layers[L].sum_var, type_=self.activation_types[L]))

            if L == 0:
                self.layers[L].compute_dl_dw(x0)
            else:
                self.layers[L].compute_dl_dw(self.layers[L - 1].x_var) 

            self.layers[L].compute_dl_db()

    def zero_grad(self):
        for layer in self.layers:
            layer.set_zero_grad()

    def predicted_values(self):
        return self.layers[self.nb_layers-1].x_var

def main():
    train_input, train_target, test_input, test_target = prologue.load_data(one_hot_labels=True, normalize=True)
    print('train_input', train_input.size(), 'train_target', train_target.size())
    print('test_input', test_input.size(), 'test_target', test_target.size())

    train_target = train_target * 0.9

    mini_batch_size = 100#train_input.size(0)
    nb_features_layer = [50, 10]
    activation_types = ["tanh", "tanh"]
    nb_layers = 2 # nb_layers - 1 hidden layers, the last being the prediction. 

    # Initialize the model
    model = Net(nb_layers, nb_features_layer, activation_types, train_input.shape[-1])

    #SGD step
    eta = 0.1/1000
    for epoch in range(10):
        total_loss_epoch = 0

        for b in range(0, train_input.size(0), mini_batch_size):
            # Set to zero the gradients
            model.zero_grad()
            # Perform the forward pass on the mini batch
            model.forward_pass(train_input.narrow(0, b, mini_batch_size))
            # Perform the backward pass on the mini batch
            model.backward_pass(train_input.narrow(0, b, mini_batch_size), train_target.narrow(0, b, mini_batch_size))

            # Update the weights and the biases
            for L in range(nb_layers):
                ### Should we do a function that modifies these values? Or is it safe to modify the
                ### inner variables of a class?
                model.layers[L].weights -= eta * model.layers[L].dl_dw_tot
                model.layers[L].bias -= eta * model.layers[L].dl_db_tot

        for b in range(0, train_input.size(0), mini_batch_size):
            total_loss_epoch += loss(model.predicted_values(), train_target.narrow(0, b, mini_batch_size))

        print(f"Epoch {epoch}: loss = {total_loss_epoch}")

    model.forward_pass(train_input)    
    print(f"Loss with the whole set: {loss(model.predicted_values(), train_target)}")

    total_loss_epoch = 0 
    for b in range(0, train_input.size(0), mini_batch_size):
        model.forward_pass(train_input.narrow(0, b, mini_batch_size))    
        total_loss_epoch += loss(model.predicted_values(), train_target.narrow(0, b, mini_batch_size))

    print(f"Sum of all batches' losses: Loss = {total_loss_epoch}")


if __name__ == '__main__':
    main()
