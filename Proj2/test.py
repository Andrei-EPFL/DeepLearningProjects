from torch import empty
from torch import manual_seed, set_grad_enabled
from torch import set_default_dtype, set_printoptions, float64, float32
from torch import load

import math
import time

import dl

set_printoptions(precision=5)
set_default_dtype(float64)
set_grad_enabled(False)


class Net(dl.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = dl.Linear(2, 25)
        self.relu1 = dl.ReLU()
        self.fc2 = dl.Linear(25, 25)
        self.relu2 = dl.ReLU()
        self.fc3 = dl.Linear(25, 25)
        self.relu3 = dl.ReLU()
        self.fc4 = dl.Linear(25, 2)
        self.sigmoid = dl.Sigmoid()
                         
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

    def param(self):
        params = []
        for key, module in self.__dict__.items():
            try:
                params += module.param()
            except:
                continue
        return params

def generate_disc_set(nb, one_hot_encode=True):
    ''' 
        Generate the data set:
        All points inside a radius of sqrt( 1 / (2 * PI) ) have
        the label equal to 1 and all outside points have the
        label equal to 0. 
    '''
    data = empty(size=(nb, 2)).uniform_(0, 1)
    radius = (data - 0.5).pow(2).sum(axis=1)
    labels = (radius < 1./(2 * math.pi)).long()
    if one_hot_encode:
        out = empty(size=(data.shape[0], 2)).fill_(0)
        out[~labels.bool(),0] = 1
        out[labels.bool(),1] = 1
        return data, out, labels
    else:
        return data, labels

if __name__ == '__main__':
    manual_seed(42)
    batch_size = 10
    epochs = 50
    learning_rate = 5e-1
    
    ### Generate the data set: train and validation sets
    # train_input, train_target, train_labels = generate_disc_set(1000, one_hot_encode=True)
    # validation_input, validation_target, validation_labels = generate_disc_set(1000, one_hot_encode=True)

    train_input = load("./data/train_input_float32_S42.pt").double()
    train_target = load("./data/train_target_float32_S42.pt").double()
    train_labels = load("./data/train_labels_float32_S42.pt").double()
    
    validation_input = load("./data/validation_input_float32_S42.pt").double()
    validation_target = load("./data/validation_target_float32_S42.pt").double()
    validation_labels = load("./data/validation_labels_float32_S42.pt").double()

    print(f"Number in: {train_labels.sum()}, Number out: {1000 - train_labels.sum()}")

    ### Define the model
    # model = Net()
    model = dl.Sequential(dl.Linear(2, 25),
                           dl.ReLU(),
                           dl.Linear(25, 25),
                           dl.ReLU(),
                           dl.Linear(25, 25),
                           dl.ReLU(),
                           dl.Linear(25, 2),
                           dl.Sigmoid()
                        )

    ### Define the loss
    criterion = dl.LossMSE()

    ### Start training for n number of epochs
    val_losses = []
    val_accuracies = []
    start_time = time.time()
    for e in range(epochs):

        train_losses = []
        train_accuracies = []
        
        ### Split the data set in mini batches
        n_batches = train_input.shape[0] // batch_size
        for batch in range(0, train_input.shape[0], batch_size):

            ### Call the forward pass
            out = model(dl.nTensor(tensor=train_input.narrow(0, batch, batch_size)))
            ### Compute the loss
            train_loss = criterion(out, dl.nTensor(tensor=train_target.narrow(0, batch, batch_size)))
            train_losses.append(train_loss.tensor.item())
            
            ### Compute the accuracy
            train_accuracy = (out.tensor.argmax(axis=1) == train_target.narrow(0, batch, batch_size).argmax(axis=1)).float().mean()
            train_accuracies.append(train_accuracy.item())
            
            ### Set all gradients of the parameters to zero.
            model.zero_grad()
            
            ### Call the backward pass (two possible methods)
            train_loss.backward()
            # model.backward()

            ### Update the parameters
            for param in model.param():
                param.tensor-= learning_rate * param.grad

        ### Validation step after the end of training in one epoch
        out = model(dl.nTensor(tensor=validation_input))
        val_loss = criterion(out, dl.nTensor(tensor=validation_target))
        val_losses.append(val_loss.tensor.item())
        val_accuracy = (out.tensor.argmax(axis=1) == validation_target.argmax(axis=1)).float().mean()
        val_accuracies.append(val_accuracy.item())

        if e % 1 == 0 or e == epochs - 1:
            print(f"Epoch {e}: ")
            print(f"\tTrain loss: {sum(train_losses) / n_batches:.20e}\t Train acc: {sum(train_accuracies) / n_batches:.20f}")
            print(f"\tVal loss: {val_loss.tensor.item():.20e}\t Val acc: {val_accuracy.item():.20f}")

    print(f"\n==> End of training after {time.time()-start_time} seconds. Generating a new test set\n", flush=True)

    ### Generate a new test set and recompute the accuracy and the loss
    # test_input, test_target, test_labels = generate_disc_set(1000, one_hot_encode=True)
    
    test_input=load("./data/test_input_float32_S42.pt").double()
    test_target=load("./data/test_target_float32_S42.pt").double()
    test_labels=load("./data/test_labels_float32_S42.pt").double()
    
    out = model(dl.nTensor(tensor=test_input))
    test_loss = criterion(out, dl.nTensor(tensor=test_target))
    out_labels = out.tensor.argmax(axis=1)
    test_accuracy = (out_labels == test_target.argmax(axis=1)).float().mean()
    test_err = 1 - test_accuracy

    print(f"Final test loss: {test_loss.tensor.item():.3f}\tFinal test acc: {test_accuracy:.2f}\tFinal test error {test_err:.2f}")
    
    ### Write the positions of points the true labels and the predicted labels
    outfile = open("results/float64_dl_test_output_S42.dat", 'w')
    for i in range(len(test_input)):
        outfile.write(f"{test_input[i,0]} {test_input[i,1]} {out_labels[i]} {test_labels[i]}\n")
    outfile.close()