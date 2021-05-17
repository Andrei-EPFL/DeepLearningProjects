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
    tanh = dl.Tanh()
    criterion = dl.LossMSE()

    lin1 = dl.Linear(2, 25)
    lin2 = dl.Linear(25, 2)
    
    input = dl.nTensor(tensor=train_input)
    # input.backward()
    # output1 = relu(input)
    output1 = relu(lin1(input))
    # output1 = lin1(input)
    # output1 = tanh(input)
    output2 = (lin2(output1))
    
    # output1.backward()
    # print(output1.backward().tensor, "\n\n")
    loss = criterion(output2, dl.nTensor(tensor=train_target))
    
    lsb = loss.backward()
    #print(lsb.tensor)
    #print((loss.backward().tensor))
    print(input.grad)

def pytorch_implem(train_input, train_target, train_labels):
    torch.set_grad_enabled(True)

    print("!!Pytorch implementation!!")

    relu = torch.nn.ReLU()
    tanh = torch.nn.Tanh()

    criterion = torch.nn.MSELoss()

    lin1 = torch.nn.Linear(2, 25)
    lin2 = torch.nn.Linear(25, 2)
    input = torch.autograd.Variable(train_input, requires_grad=True)
    # input.backward(torch.ones_like(input))
    
    #output1 = relu(input)
    output1 =relu(lin1(input))
    # output1 = lin1(input)
    # output1 = tanh(input)
    output2 = (lin2(output1))

    # print(output1.backward(torch.ones_like(output1)))
    loss = criterion(output2, train_target)
    print(loss.backward())
    print(1000*input.grad)


if __name__ == '__main__':
    torch.manual_seed(42)
    batch_size = 10
    epochs = 200
    learning_rate = 5e-1
    
    train_input, train_target, train_labels = generate_disc_set(1000, one_hot_encode=True)
    test_input, test_target, test_labels = generate_disc_set(1000, one_hot_encode=True)
    import sys
    if sys.argv[1] == '0':
        our_implem(train_input, train_target, train_labels)
    elif sys.argv[1] == '1':
        pytorch_implem(train_input, train_target, train_labels)
    else:
        print("USAGE: python test_backward_pass.py <<implementation>>, where <<implementation>> is 0 for our framework and 1 for pytorch")
