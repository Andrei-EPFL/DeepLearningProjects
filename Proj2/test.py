from torch import empty, manual_seed
import math

from dl import dl

def generate_disc_set(nb, one_hot_encode=True):
    data = empty((nb, 2)).uniform_(0, 1)
    radius = (data - 0.5).pow(2).sum(axis=1)
    labels = (radius < 2./math.pi).long()
    if one_hot_encode:
        out = empty((data.shape[0], 2)).fill_(0).long()
        out[:,labels] = 1
        return data, out
    else:
        return data, labels

if __name__ == '__main__':

    manual_seed(42)
    batch_size = 500
    epochs = 10
    learning_rate = 1e-3

    train_input, train_target = generate_disc_set(1000)
    test_input, test_target = generate_disc_set(1000)

    model = dl.Sequential(dl.Linear(2, 25, batch_size),
                          dl.ReLU(),
                          dl.Linear(25, 25, batch_size),
                          dl.ReLU(),
                          dl.Linear(25, 25, batch_size),
                          dl.ReLU(),
                          dl.Linear(25, 25, batch_size),
                          dl.ReLU(),
                          dl.Linear(25, 2, batch_size),
                          dl.ReLU())
    
    criterion = dl.LossMSE()
    model.summary()
    outputs = []
    for e in range(epochs):

        train_losses = []
        train_accuracies = []
        
        for batch in range(0, train_input.shape[0], batch_size):
            out = model(train_input.narrow(0, batch, batch_size))
            outputs.append(out)
            train_loss = criterion(out, train_target.narrow(0, batch, batch_size))

            train_accuracy = (out.argmax(axis=1) == train_target.narrow(0, batch, batch_size).argmax(axis=1)).float().mean()
            

            model.zero_grad()
            model.backward(criterion.backward())

            for param in model.param():
                #old_param = param.clone()
                
                param = param - learning_rate * param.grad
                #print((old_param - param).mean())
                pass


            print(train_loss, train_accuracy)


