from torch import empty, manual_seed
import math

from dl import dl

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
        self.input = input
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        self.output = x
        return self.output

    def param(self):
        params = []
        for key, module in self.__dict__.items():
            try:
                params += module.param()
            except:
                continue
        return params

def generate_disc_set(nb, one_hot_encode=True):
    data = dl.nTensor(tensor=empty(size=(nb, 2)).uniform_(0, 1))

    radius = dl.nTensor(tensor=(data() - 0.5).pow(2).sum(axis=1))
    labels = dl.nTensor(tensor=(radius() < 1./(2 * math.pi)).long())
    if one_hot_encode:
        out = dl.nTensor(tensor=empty(size=(data().shape[0], 2)).fill_(0).float())

        out.tensor[~labels().bool(),0] = 1
        out.tensor[labels().bool(),1] = 1
        return data, out, labels
    else:
        return data, labels

if __name__ == '__main__':
    manual_seed(42)
    batch_size = 10
    epochs = 50
    learning_rate = 5e-1

    # import torch
    # test_torch = torch.tensor([2], dtype=torch.float32)
    # print(test_torch)
    # print(type(test_torch))
    # print(empty(0).__class__)
    test = dl.nTensor([[2,3]])
    print(type(test))
    print(test)
    print(test.shape)
    print(test.created_by)
    exit()

    train_input, train_target, train_labels = generate_disc_set(1000, one_hot_encode=True)
    test_input, test_target, test_labels = generate_disc_set(1000, one_hot_encode=True)
    
    #print(type(train_input))
    #exit()
    model = Net()
    # model = dl.Sequential(dl.Linear(2, 25),
    #                       dl.ReLU(),
    #                       dl.Linear(25, 25),
    #                       dl.ReLU(),
    #                       dl.Linear(25, 25),
    #                       dl.ReLU(),
    #                       dl.Linear(25, 25),
    #                       dl.ReLU(),
    #                       dl.Linear(25, 2),
    #                       dl.Sigmoid())
    
    criterion = dl.LossMSE()
    
    
    
    val_losses = []
    val_accuracies = []
    for e in range(epochs):

        train_losses = []
        train_accuracies = []
        
        n_batches = train_input().shape[0] // batch_size
        for batch in range(0, train_input().shape[0], batch_size):
            out = model(dl.nTensor(tensor=train_input().narrow(0, batch, batch_size)))
            train_loss = criterion(out, dl.nTensor(tensor=train_target().narrow(0, batch, batch_size)))
            train_losses.append(train_loss().item())
            train_accuracy = (out().argmax(axis=1) == train_target().narrow(0, batch, batch_size).argmax(axis=1)).float().mean()
            train_accuracies.append(train_accuracy.item())
            

            model.zero_grad()

            #model.backward(criterion.backward_())
            train_loss.backward()

            for param in model.param():
                param.tensor-= learning_rate * param().grad
                
                
            
            #print(train_loss, train_accuracy)

        # Validation

        out = model(test_input)
        val_loss = criterion(out, test_target)
        val_losses.append(val_loss().item())
        val_accuracy = (out().argmax(axis=1) == test_target().argmax(axis=1)).float().mean()
        val_accuracies.append(val_accuracy.item())

        if e % 10 == 0:
            print(f"Epoch {e}: ")
            print(f"\tTrain loss: {sum(train_losses) / n_batches:.2e}\t Train acc: {sum(train_accuracies) / n_batches:.2f}")
            print(f"\tVal loss: {val_loss().item():.2e}\t Val acc: {val_accuracy.item():.2f}")


