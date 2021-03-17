#!/usr/bin/env python
import torch


class NN(torch.nn.Module):

    def __init__(self, out_channels_1=32, out_channels_2=64, kernel_size_1=5, kernel_size_2=3, nhidden=200):
        super().__init__()

        self.out_channels_1 = out_channels_1
        self.out_channels_2 = out_channels_2
        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2
        self.nhidden = nhidden

        # Define layers for network
        
        # A ConvNet to identify the digits
        self.conv1 = torch.nn.Conv2d(1, self.out_channels_1, self.kernel_size_1, stride=1, padding=0)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(self.out_channels_1, self.out_channels_2, self.kernel_size_2, stride=1, padding=0)
        self.dense = torch.nn.Linear(in_features = self.out_channels_2*3*3, out_features = self.nhidden)
        self.denseout = torch.nn.Linear(in_features = self.nhidden, out_features = 10)
        self.relu = torch.nn.ReLU()

        # A dense network to know which is larger
        self.dense1 = torch.nn.Linear(in_features = 20, out_features = 32)
        self.dense2 = torch.nn.Linear(in_features = 32, out_features = 64)
        self.dense3 = torch.nn.Linear(in_features = 64, out_features = 2)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, input):
        in_1 = input[:, 0, :, :].unsqueeze(1)
        in_2 = input[:, 1, :, :].unsqueeze(1)
        out_1 = self.conv1(in_1)
        out_1 = self.relu(self.maxpool1(out_1))
        out_1 = self.relu(self.conv2(out_1))
        out_1 = self.dense(out_1.view(out_1.shape[0], -1))
        out_1 = self.denseout(out_1)

        out_2 = self.conv1(in_2)
        out_2 = self.relu(self.maxpool1(out_2))
        out_2 = self.relu(self.conv2(out_2))
        out_2 = self.dense(out_2.view(out_2.shape[0], -1))
        out_2 = self.denseout(out_2)

        out = self.dense1(torch.cat((out_1, out_2), dim=1))
        out = self.relu(self.dense2(out))
        out = self.softmax(self.dense3(out))

        return out_1, out_2, out



def train(model, train_input, train_target, train_classes,
            n_epochs, batch_size, device, validation_fraction=0.5, learning_rate=1e-3):
        
        
        #torch.manual_seed(seed)
        criterion_classes = torch.nn.CrossEntropyLoss().to(device)
        criterion_label = torch.nn.CrossEntropyLoss().to(device)
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

        for epoch in range(n_epochs):

            train_losses = []
            train_accuracies = []
            for batch in range(0, train_input.shape[0], batch_size):
                out_1, out_2, out = model(train_input.narrow(0, batch, batch_size))
                loss_1 = criterion_classes(out_1, train_classes.narrow(0, batch, batch_size)[:,0])
                loss_2 = criterion_classes(out_2, train_classes.narrow(0, batch, batch_size)[:,1])
                loss_out = criterion_label(out, train_target.narrow(0, batch, batch_size))

                out_class = torch.argmax(out, axis=1).to(int)
                
                accuracy = (out_class == train_target.narrow(0, batch, batch_size)).to(float).mean()
                train_accuracies.append(accuracy.item())


                loss = loss_out + loss_1 + loss_2
                train_losses.append(loss.item())
                model.zero_grad()
                loss.backward()
                optimizer.step()
            
            
            with torch.no_grad():
                out_1, out_2, out = model(validation_input)
                loss_1 = criterion_classes(out_1, validation_classes[:,0])
                loss_2 = criterion_classes(out_2, validation_classes[:,1])
                loss_out = criterion_label(out, validation_target)

                val_loss = (loss_out + loss_1 + loss_2).item()
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

    
    train_input, train_target, train_classes, test_imput, test_target, test_classes = generate_pair_sets(5000)
    model = NN()
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