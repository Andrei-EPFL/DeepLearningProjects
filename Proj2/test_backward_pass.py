import torch

import math

def generate_disc_set(nb, one_hot_encode=True):
    data = torch.empty(size=(nb, 2)).uniform_(0, 1)
    radius = (data - 0.5).pow(2).sum(axis=1)
    labels = (radius < 1./(2 * math.pi)).long()
    if one_hot_encode:
        out = torch.empty(size=(data.shape[0], 2)).fill_(0).float()
        out[~labels.bool(),0] = 1
        out[labels.bool(),1] = 1
        return data, out, labels
    else:
        return data, labels


def our_implem(train_input, train_target, train_labels):
    torch.set_grad_enabled(False)
    import dl

    print("!!Our implementation!!")
    
    relu = dl.ReLU()
    criterion = dl.LossMSE()

    lin1 = dl.Linear(2, 25)
    lin2 = dl.Linear(25, 2)
    
    input = dl.nTensor(tensor=train_input)
    
    output1 = lin1(input)
    output2 = relu(lin2(output1))
    
    print(output2.backward().tensor, "\n\n")
    #loss = criterion(output, dl.nTensor(tensor=train_target))
    #print((loss.backward().tensor))
    print(input.grad.tensor)

def pytorch_implem(train_input, train_target, train_labels):
    torch.set_grad_enabled(True)

    print("!!Pytorch implementation!!")

    relu = torch.nn.ReLU()
    criterion = torch.nn.MSELoss()

    lin1 = torch.nn.Linear(2, 25)
    lin2 = torch.nn.Linear(25, 2)
    input = torch.autograd.Variable(train_input, requires_grad=True)
    
    output1 = lin1(input)
    output2 = relu(lin2(output1))

    print(output2.backward(torch.ones_like(output2)))
    #loss = criterion(output, train_target)
    #print(loss.backward())
    print(input.grad)


if __name__ == '__main__':
    torch.manual_seed(42)
    batch_size = 10
    epochs = 200
    learning_rate = 5e-1
    
    train_input, train_target, train_labels = generate_disc_set(1000, one_hot_encode=True)
    test_input, test_target, test_labels = generate_disc_set(1000, one_hot_encode=True)

    #our_implem(train_input, train_target, train_labels)
    pytorch_implem(train_input, train_target, train_labels)