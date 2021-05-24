import torch
import math
import time

torch.set_printoptions(precision=30)
torch.set_default_dtype(torch.float64)
def generate_disc_set(nb, one_hot_encode=True):
    ''' 
        Generate the data set:
        All points inside a radius of sqrt( 1 / (2 * PI) ) have
        the label equal to 1 and all outside points have the
        label equal to 0. 
    '''
    data = torch.empty((nb, 2)).uniform_(0, 1)
    radius = (data - 0.5).pow(2).sum(axis=1)
    labels = (radius < 1./(2 * math.pi)).long()
    if one_hot_encode:
        out = torch.empty((data.shape[0], 2)).fill_(0)
        out[~labels.bool(),0] = 1
        out[labels.bool(),1] = 1
        return data, out, labels
    else:
        return data, labels


if __name__ == '__main__':

    torch.manual_seed(42)
    batch_size = 10
    epochs = 50
    learning_rate = 5e-1

    ### Generate the data set: train and validation sets
    # train_input, train_target, train_labels = generate_disc_set(1000, one_hot_encode=True)
    # validation_input, validation_target, validation_labels = generate_disc_set(1000, one_hot_encode=True)
    
    train_input = torch.load("./data/train_input_float32_S42.pt").double()
    train_target = torch.load("./data/train_target_float32_S42.pt").double()
    train_labels = torch.load("./data/train_labels_float32_S42.pt").double()
    
    validation_input = torch.load("./data/validation_input_float32_S42.pt").double()
    validation_target = torch.load("./data/validation_target_float32_S42.pt").double()
    validation_labels = torch.load("./data/validation_labels_float32_S42.pt").double()

    print(f"Number in: {train_labels.sum()}, Number out: {1000 - train_labels.sum()}")
    
    ### Define the model
    model = torch.nn.Sequential(torch.nn.Linear(2, 25),
                          torch.nn.ReLU(),
                          torch.nn.Linear(25, 25),
                          torch.nn.ReLU(),
                          torch.nn.Linear(25, 25),
                          torch.nn.ReLU(),
                          torch.nn.Linear(25, 2),
                          torch.nn.Tanh()
                        )

    ### Define the loss
    criterion = torch.nn.MSELoss()

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
            out = model(train_input.narrow(0, batch, batch_size))
            
            ### Compute the loss
            train_loss = criterion(out, train_target.narrow(0, batch, batch_size))
            train_losses.append(train_loss.item())
            
            ### Compute the accuracy
            train_accuracy = (out.argmax(axis=1) == train_target.narrow(0, batch, batch_size).argmax(axis=1)).float().mean()
            train_accuracies.append(train_accuracy.item())
            
            ### Set all gradients of the parameters to zero.
            model.zero_grad()
            
            ### Call the backward pass
            train_loss.backward()

            ### Update the parameters
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad
                    
        ### Validation step after the end of training in one epoch
        out = model(validation_input)
        val_loss = criterion(out, validation_target)
        val_losses.append(val_loss.item())
        val_accuracy = (out.argmax(axis=1) == validation_target.argmax(axis=1)).float().mean()
        val_accuracies.append(val_accuracy.item())

        if e % 1 == 0 or e == epochs - 1:
            print(f"Epoch {e}: ")
            print(f"\tTrain loss: {sum(train_losses) / n_batches:.20e}\t Train acc: {sum(train_accuracies) / n_batches:.20f}")
            print(f"\tVal loss: {val_loss.item():.20e}\t Val acc: {val_accuracy.item():.20f}")

    print(f"\n==> End of training after {time.time()-start_time} seconds. Generating a new test set\n", flush=True)

    ### Generate a new test set and recompute the accuracy and the loss
    # test_input, test_target, test_labels = generate_disc_set(1000, one_hot_encode=True)
    
    test_input=torch.load("./data/test_input_float32_S42.pt").double()
    test_target=torch.load("./data/test_target_float32_S42.pt").double()
    test_labels=torch.load("./data/test_labels_float32_S42.pt").double()
    
    out = model(test_input)
    test_loss = criterion(out, test_target)
    out_labels = out.argmax(axis=1)
    test_accuracy = (out_labels == test_target.argmax(axis=1)).float().mean()
    test_err = 1-test_accuracy

    print(f"Final test loss: {test_loss.item():.3f}\tFinal test acc: {test_accuracy:.2f}\tFinal test error {test_err:.2f}")
    
    ### Write the positions of points the true labels and the predicted labels
    outfile = open("results/float64_pt_test_output_S42.dat", 'w')
    for i in range(len(test_input)):
        outfile.write(f"{test_input[i,0]} {test_input[i,1]} {out_labels[i]} {test_labels[i]}\n")
    outfile.close()

    ##########################
    ##########################


    ### Attempt to generate an adversarial example
    msecriterion = torch.nn.MSELoss()
    index = 0
    in_n = torch.empty(2)
    in_n.copy_(train_input[index])
    tar_n = train_target[index]
    in_n.requires_grad_()

    lr = 0.1
    optimizer = torch.optim.SGD([in_n], lr=lr)
    for k in range(20):
        out = model(in_n)
        loss = - msecriterion(out, tar_n)

        print(f"\nStep={k}: loss={loss}")
        optimizer.zero_grad()
        loss.backward()
        print(" Input before update: ", in_n)

        optimizer.step()
        print("Input after update: ", in_n)
        print("Gradient with respect the input: ", in_n.grad)


    tensor_A1 = torch.empty(2)
    tensor_A1[0], tensor_A1[1] = 0.4264, 0.8062
    tensor_A2 = in_n

    print(f"\nInitial position: {train_input[index]}; Label: {train_target[index]}; Radius: {(train_input[index] - 0.5).pow(2).sum()}/{1./(2*math.pi)}")

    print(f"dl Adv ex: {tensor_A1}; OutLabel: {model(tensor_A1)}; Radius: {(tensor_A1 - 0.5).pow(2).sum()}/{1./(2*math.pi)}")
    print(f"PyTorch Adv ex: {tensor_A2}; OutLabel: {model(tensor_A2)}; Radius: {(tensor_A2 - 0.5).pow(2).sum()}/{1./(2*math.pi)}")
