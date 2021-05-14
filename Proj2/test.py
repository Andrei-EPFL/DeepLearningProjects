from torch import empty, manual_seed, set_grad_enabled
import math

import dl

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
        self.fc4 = dl.Linear(25, 25)
        self.relu4 = dl.ReLU()
        self.fc5 = dl.Linear(25, 2)
        self.sigmoid = dl.Sigmoid()
                         
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
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
    data = empty(size=(nb, 2)).uniform_(0, 1)
    radius = (data - 0.5).pow(2).sum(axis=1)
    labels = (radius < 1./(2 * math.pi)).long()
    if one_hot_encode:
        out = empty(size=(data.shape[0], 2)).fill_(0).float()
        out[~labels.bool(),0] = 1
        out[labels.bool(),1] = 1
        return data, out, labels
    else:
        return data, labels

if __name__ == '__main__':
    manual_seed(42)
    batch_size = 1000
    epochs = 200
    learning_rate = 5e-1
    
    train_input, train_target, train_labels = generate_disc_set(1000, one_hot_encode=True)
    test_input, test_target, test_labels = generate_disc_set(1000, one_hot_encode=True)
    # model = Net()

    model = dl.Sequential(dl.Linear(2, 25),
                           dl.ReLU(),
                           dl.Linear(25, 25),
                           dl.ReLU(),
                           dl.Linear(25, 25),
                           dl.ReLU(),
                           dl.Linear(25, 25),
                           dl.ReLU(),
                           dl.Linear(25, 2)
                        #    dl.Sigmoid()
                        )

    criterion = dl.LossMSE()
  
    val_losses = []
    val_accuracies = []
    for e in range(epochs):

        train_losses = []
        train_accuracies = []
        
        n_batches = train_input.shape[0] // batch_size
        for batch in range(0, train_input.shape[0], batch_size):
            out = model(dl.nTensor(tensor=train_input.narrow(0, batch, batch_size)))
            
            train_loss = criterion(out, dl.nTensor(tensor=train_target.narrow(0, batch, batch_size)))
            train_losses.append(train_loss.tensor.item())
            
            train_accuracy = (out.tensor.argmax(axis=1) == train_target.narrow(0, batch, batch_size).argmax(axis=1)).float().mean()
            train_accuracies.append(train_accuracy.item())
            
            
            model.zero_grad()
            
            train_loss.backward()
            #model.backward(criterion.backward())
            for param in model.param():
                param.tensor-= learning_rate * param.grad
                
                
            
            #print(train_loss, train_accuracy)

        # Validation

        out = model(dl.nTensor(tensor=test_input))
        val_loss = criterion(out, dl.nTensor(tensor=test_target))
        val_losses.append(val_loss.tensor.item())
        val_accuracy = (out.tensor.argmax(axis=1) == test_target.argmax(axis=1)).float().mean()
        val_accuracies.append(val_accuracy.item())

        if e % 10 == 0:
            print(f"Epoch {e}: ")
            print(f"\tTrain loss: {sum(train_losses) / n_batches:.20e}\t Train acc: {sum(train_accuracies) / n_batches:.20f}")
            print(f"\tVal loss: {val_loss.tensor.item():.20e}\t Val acc: {val_accuracy.item():.20f}")

    print(f"==> End of training, generating a new test set", flush=True)

    test_input, test_target, test_labels = generate_disc_set(1000, one_hot_encode=True)
    out = model(dl.nTensor(tensor=test_input))
    test_loss = criterion(out, dl.nTensor(tensor=test_target))
    out_labels = out.tensor.argmax(axis=1)
    test_accuracy = (out_labels == test_target.argmax(axis=1)).float().mean()
    test_err = 1-test_accuracy

    print(f"Final test loss: {test_loss.tensor.item():.3f}\tFinal test acc: {test_accuracy:.2f}\tFinal test error {test_err:.2f}")
    outfile = open("results/test_output.dat", 'w')
    for i in range(len(test_input)):
        outfile.write(f"{test_input[i,0]} {test_input[i,1]} {out_labels[i]} {test_labels[i]}\n")
    outfile.close()


    mseloss = dl.LossMSE()

    in_  = train_input[2:4]
    tar_ = train_target[2:4]
    if len(in_.shape) == 1:
        length = 1
    else:
        length = len(in_)

    lr = 0.1
    in_n = dl.nTensor(tensor=in_)

    for k in range(15):
        out = model(in_n)
        loss = mseloss(out, dl.nTensor(tensor=tar_))
        model.zero_grad()
        loss.backward()
        print(k, " Before: ", in_n.tensor)
        in_n.tensor = in_n.tensor + lr * in_n.grad / length
        print("After: ",in_n.tensor)
        print("Grad: ", in_n.grad / length)
        print("\n")

