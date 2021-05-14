import torch
import math
import matplotlib.pyplot as plt

def generate_disc_set(nb, one_hot_encode=True):
    data = torch.empty((nb, 2)).uniform_(0, 1)
    radius = (data - 0.5).pow(2).sum(axis=1)
    labels = (radius < 1./(2 * math.pi)).long()
    if one_hot_encode:
        out = torch.empty((data.shape[0], 2)).fill_(0).float()
        out[~labels.bool(),0] = 1
        out[labels.bool(),1] = 1
        return data, out, labels
    else:
        return data, labels


if __name__ == '__main__':

    torch.manual_seed(42)
    batch_size = 100
    epochs = 30
    learning_rate = 5e-1

    train_input, train_target, train_labels = generate_disc_set(1000, one_hot_encode=True)
    test_input, test_target, test_labels = generate_disc_set(1000, one_hot_encode=True)
    
    #plt.scatter(train_input[train_labels.bool(),0], train_input[train_labels.bool(),1], c='r')
    #plt.scatter(train_input[~train_labels.bool(),0], train_input[~train_labels.bool(),1], c='k')
    ##plt.show()
    print(f"Number in: {train_labels.sum()}, Number out: {1000 - train_labels.sum()}")
    #exit()

    model = torch.nn.Sequential(torch.nn.Linear(2, 25),
                          torch.nn.ReLU(),
                          torch.nn.Linear(25, 25),
                          torch.nn.ReLU(),
                          torch.nn.Linear(25, 25),
                          torch.nn.ReLU(),
                          torch.nn.Linear(25, 25),
                          torch.nn.ReLU(),
                          torch.nn.Linear(25, 2))
    
    criterion = torch.nn.MSELoss()

    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    for e in range(epochs):

        train_losses = []
        train_accuracies = []
        
        for batch in range(0, train_input.shape[0], batch_size):
            out = model(train_input.narrow(0, batch, batch_size))
            
            train_loss = criterion(out, train_target.narrow(0, batch, batch_size))

            train_accuracy = (out.argmax(axis=1) == train_target.narrow(0, batch, batch_size).argmax(axis=1)).float().mean()
            train_accuracies.append(train_accuracy.item())
            
            model.zero_grad()
            train_loss.backward()
            #optimizer.step()
            #optimizer.zero_grad()
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad
                    
            
            #print(train_loss, train_accuracy)
        print(torch.Tensor(train_accuracies).mean())

