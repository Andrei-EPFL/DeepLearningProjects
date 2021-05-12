#!/usr/bin/env python
import torch

def train(model, train_input, train_target, train_classes,
            n_epochs, batch_size, device, validation_fraction=0.5, learning_rate=1e-3, use_aux_loss = True):
        
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_total_params_rg = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Training model of class: {model.__class__.__name__}")
        print(f"The model has: {pytorch_total_params} params and {pytorch_total_params_rg} params that require grad.")
    
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
                ### here shouldn't the model be in evaluation mode to stop the pool layers?
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