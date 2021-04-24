#!/usr/bin/env python
import torch
class ConvNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels_1=32, 
                out_channels_2=64, kernel_size_1=5,
                kernel_size_2=3, n_hidden=100, n_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels_1 = out_channels_1
        self.out_channels_2 = out_channels_2
        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.dummy = torch.zeros((1, self.in_channels, 14, 14))
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(in_channels, self.out_channels_1, self.kernel_size_1, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2),
                                        torch.nn.Conv2d(self.out_channels_1, self.out_channels_2, self.kernel_size_2, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2))
        dummy_out = self.conv(self.dummy)
        n_elem = dummy_out.shape[-1] * dummy_out.shape[-2] * dummy_out.shape[-3]
        self.dense = torch.nn.Sequential(torch.nn.Linear(in_features = n_elem, out_features = self.n_hidden),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features = self.n_hidden, out_features = self.n_classes))

    def forward(self, x):
        out = self.conv(x)
        out = self.dense(out.view(out.shape[0], -1))
        return out



class NN(torch.nn.Module):

    def __init__(self, out_channels_1=32, out_channels_2=64, kernel_size_1=5, kernel_size_2=3, n_hidden=200):
        super().__init__()

        self.out_channels_1 = out_channels_1
        self.out_channels_2 = out_channels_2
        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2
        self.n_hidden = n_hidden

        # Define layers for network
        
        # Two identical ConvNets to avoid weight sharing
        self.conv1 = ConvNet(in_channels=1, n_classes=10, 
                            out_channels_1=self.out_channels_1,
                            out_channels_2=self.out_channels_2,
                            kernel_size_1=self.kernel_size_1,
                            kernel_size_2=self.kernel_size_2,
                            n_hidden=self.n_hidden)
        self.conv2 = ConvNet(in_channels=1, n_classes=10, 
                            out_channels_1=self.out_channels_1,
                            out_channels_2=self.out_channels_2,
                            kernel_size_1=self.kernel_size_1,
                            kernel_size_2=self.kernel_size_2,
                            n_hidden=self.n_hidden)
        

        self.dense = torch.nn.Sequential(torch.nn.Linear(in_features = 20, out_features = 32),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features = 32, out_features = 64),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features = 64, out_features = 2),
                                        torch.nn.Softmax(dim=1))



    def forward(self, input):
        in_1 = input[:, 0, :, :].unsqueeze(1)
        in_2 = input[:, 1, :, :].unsqueeze(1)
        out_1 = self.conv1(in_1)
        out_2 = self.conv2(in_2)
        out = self.dense(torch.cat((out_1, out_2), dim=1))
        return out_1, out_2, out

class NN_ws(torch.nn.Module):

    def __init__(self, out_channels_1=32, out_channels_2=64, kernel_size_1=5, kernel_size_2=3, n_hidden=200):
        super().__init__()

        self.out_channels_1 = out_channels_1
        self.out_channels_2 = out_channels_2
        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2
        self.n_hidden = n_hidden

        # Define layers for network
        
        # A ConvNet to identify the digits
        self.conv = ConvNet(in_channels=1, n_classes=10, 
                            out_channels_1=self.out_channels_1,
                            out_channels_2=self.out_channels_2,
                            kernel_size_1=self.kernel_size_1,
                            kernel_size_2=self.kernel_size_2,
                            n_hidden=self.n_hidden)

        # A dense network to know which is larger
        self.dense = torch.nn.Sequential(torch.nn.Linear(in_features = 20, out_features = 32),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features = 32, out_features = 64),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features = 64, out_features = 2),
                                        torch.nn.Softmax(dim=1))



    def forward(self, input):
        in_1 = input[:, 0, :, :].unsqueeze(1)
        in_2 = input[:, 1, :, :].unsqueeze(1)
        out_1 = self.conv(in_1)
        out_2 = self.conv(in_2)
        out = self.dense(torch.cat((out_1, out_2), dim=1))
        return out_1, out_2, out



def train(model, train_input, train_target, train_classes,
            n_epochs, batch_size, device, validation_fraction=0.5, learning_rate=1e-3, use_aux_loss = True):
        
        
        #torch.manual_seed(seed)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, )

        shuffled_ids = torch.randperm(train_input.shape[0])
        val_cut = int(train_input.shape[0]*validation_fraction)

        validation_input = train_input[shuffled_ids][:val_cut].to(device)
        validation_target = train_target[shuffled_ids][:val_cut].to(device)
        validation_classes = train_classes[shuffled_ids][:val_cut].to(device)

        train_input = train_input[shuffled_ids][val_cut:].to(device)
        train_target = train_target[shuffled_ids][val_cut:].to(device)
        train_classes = train_classes[shuffled_ids][val_cut:].to(device)

        print(f"Train data shape {train_input.shape}")
        print(f"Validation data shape {validation_input.shape}")

        all_train_losses = []
        all_validation_losses = []

        all_train_acc = []
        all_validation_acc = []

        print(f"Training model of class: {model.__class__.__name__}")
        print(f"Training with aux loss: {use_aux_loss}.", flush=True)

        for epoch in range(n_epochs):

            train_losses = []
            train_accuracies = []
            for batch in range(0, train_input.shape[0], batch_size):
                out_1, out_2, out = model(train_input.narrow(0, batch, batch_size))

                out_class = torch.argmax(out, axis=1).to(int)
                accuracy = (out_class == train_target.narrow(0, batch, batch_size)).to(float).mean()
                train_accuracies.append(accuracy.item())

                loss_out = criterion(out, train_target.narrow(0, batch, batch_size))
                if use_aux_loss:
                    loss_1 = criterion(out_1, train_classes.narrow(0, batch, batch_size)[:,0])
                    loss_2 = criterion(out_2, train_classes.narrow(0, batch, batch_size)[:,1])
                    loss = loss_out + loss_1 + loss_2
                else:
                    loss = loss_out

                train_losses.append(loss.item())
                model.zero_grad()
                loss.backward()
                optimizer.step()
            
            
            with torch.no_grad():
                out_1, out_2, out = model(validation_input)
                loss_out = criterion(out, validation_target)
                if use_aux_loss:
                    loss_1 = criterion(out_1, validation_classes[:,0])
                    loss_2 = criterion(out_2, validation_classes[:,1])
                    val_loss = (loss_out + loss_1 + loss_2).item()
                else:
                    val_loss = loss_out.item()

                train_loss = sum(train_losses) / len(train_losses)

                out_class = torch.argmax(out, axis=1).to(int)
                val_accuracy = (out_class == validation_target).to(float).mean()
                train_accuracy = sum(train_accuracies) / len(train_accuracies)

                all_train_losses.append(train_loss)
                all_validation_losses.append(val_loss)

                all_train_acc.append(train_accuracy)
                all_validation_acc.append(val_accuracy)

                #scheduler.step(val_loss)

                if epoch % 5 == 0:

                    print(f"Epoch {epoch:d}. Train loss = {train_loss}. Val. loss = {val_loss}")
                    print(f"\t  Train acc = {train_accuracy}. Val. acc = {val_accuracy}")

        return all_train_losses, all_validation_losses, all_train_acc, all_validation_acc


def train_bline(model, train_input, train_target, train_classes,
            n_epochs, batch_size, device, validation_fraction=0.5, learning_rate=1e-3):
        
        
        #torch.manual_seed(seed)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, )

        shuffled_ids = torch.randperm(train_input.shape[0])
        val_cut = int(train_input.shape[0]*validation_fraction)

        validation_input = train_input[shuffled_ids][:val_cut].to(device)
        validation_target = train_target[shuffled_ids][:val_cut].to(device)
        validation_classes = train_classes[shuffled_ids][:val_cut].to(device)

        train_input = train_input[shuffled_ids][val_cut:].to(device)
        train_target = train_target[shuffled_ids][val_cut:].to(device)
        train_classes = train_classes[shuffled_ids][val_cut:].to(device)

        print(f"Train data shape {train_input.shape}")
        print(f"Validation data shape {validation_input.shape}")

        all_train_losses = []
        all_validation_losses = []

        all_train_acc = []
        all_validation_acc = []

        print(f"Training model of class: {model.__class__.__name__} as baseline")
        

        for epoch in range(n_epochs):

            train_losses = []
            train_accuracies = []
            for batch in range(0, train_input.shape[0], batch_size):
                out = model(train_input.narrow(0, batch, batch_size))

                out_class = torch.argmax(out, axis=1).to(int)
                accuracy = (out_class == train_target.narrow(0, batch, batch_size)).to(float).mean()
                train_accuracies.append(accuracy.item())

                loss = criterion(out, train_target.narrow(0, batch, batch_size))
                

                train_losses.append(loss.item())
                model.zero_grad()
                loss.backward()
                optimizer.step()
            
            
            with torch.no_grad():
                out = model(validation_input)
                val_loss = criterion(out, validation_target)
                train_loss = sum(train_losses) / len(train_losses)

                out_class = torch.argmax(out, axis=1).to(int)
                val_accuracy = (out_class == validation_target).to(float).mean()
                train_accuracy = sum(train_accuracies) / len(train_accuracies)

                all_train_losses.append(train_loss)
                all_validation_losses.append(val_loss)

                all_train_acc.append(train_accuracy)
                all_validation_acc.append(val_accuracy)

                #scheduler.step(val_loss)

                if epoch % 5 == 0:

                    print(f"Epoch {epoch:d}. Train loss = {train_loss}. Val. loss = {val_loss}")
                    print(f"\t  Train acc = {train_accuracy}. Val. acc = {val_accuracy}")

        return all_train_losses, all_validation_losses, all_train_acc, all_validation_acc




        

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from dlc_practical_prologue import generate_pair_sets


    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'

    
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(5000)
    
    model = NN_ws().to(device)
    n_epochs=100
    batch_size=5
    seed=42


    
    train_loss, val_loss, train_acc, val_acc = train(model, train_input, train_target, train_classes,
            n_epochs, batch_size, device, validation_fraction=0.3, learning_rate=1e-3)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(train_loss, label='train')
    ax[0].plot(val_loss, label='val')
    ax[0].set_ylabel('Loss')

    ax[1].plot(train_acc, label='train')
    ax[1].plot(val_acc, label='val')
    ax[1].set_ylabel('Accuracy')

    ax[0].legend(loc=0)
    [a.set_xlabel('Epochs') for a in ax]
    fig.savefig(f"results/plots/learning_curve.png", dpi=200)