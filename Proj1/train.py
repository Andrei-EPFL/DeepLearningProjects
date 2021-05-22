#!/usr/bin/env python
import torch

def train(model, 
          train_input, 
          train_target, 
          train_classes,
          n_epochs, 
          batch_size, 
          device, 
          validation_fraction=0.5, 
          learning_rate=1e-3, 
          use_aux_loss = True):
        
        """
        Function to train a model that supports using an auxiliary loss. 
        Uses CrossEntropyLoss (for sorting and classification) and SGD optimizer
        for training. Divides the train_input into train and validation sets.

        Parameters

        model: Module
            Pytorch model to train. Should output 3 Tensors of shapes
            (batch_size, 10), (batch_size, 10) and (batch_size, 2) 
            corresponding to the digit class predictions for each digit
            and the sorting prediction, respectively.
        train_input: Float Tensor shape (N, 2, 14, 14)
            Contains the train input data, pairs of MNIST downsampled 
            images.
        train_target: Long Tensor shape (N,)
            Contains the train input target. 1 if digit in position 0
            is less than or equal than the digit in position 2. 0 otherwise.
        train_classes: Long Tensor shape (N,2)
            Contains the classes of the digit pairs.
        n_epochs: int
            Number of epochs to train the model for.
        batch_size: int
            Number of samples per batch of training data.
        device: str
            String for device to use. Either 'cpu' or 'cuda'.
        validation_fraction: float
            Fraction of the N input samples to use for validation during
            training. Number of validation samples is computed as 
            int(N*validation_fraction).
        learning_rate: float
            Learning rate to train the model with.
        use_aux_loss: bool
            Whether to train the model 'model' using an auxiliary loss.

        Returns:
        
        all_train_losses: list len n_epochs
            List of train loss per batch.
        all_validation_losses: list len n_epochs
            List of validation loss per batch.
        all_train_acc: list len n_epochs
            List of train sorting accuracy per batch.
        all_validation_acc: list len n_epochs
            List of validation sorting accuracy per batch.
        

        """

        
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_total_params_rg = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Training model of class: {model.__class__.__name__}")
        print(f"The model has: {pytorch_total_params} params and {pytorch_total_params_rg} params that require grad.")
    
        
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
                model.train()
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
                model.eval()
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
        
        """
        Function to train a model that supports using an auxiliary loss. 
        Uses CrossEntropyLoss (for sorting and classification) and SGD optimizer
        for training. Divides the train_input into train and validation sets.

        Parameters

        model: Module
            Pytorch model to train. Should output 1 Tensor of shape
            (batch_size, 2) corresponding to the sorting prediction.
        train_input: Float Tensor shape (N, 2, 14, 14)
            Contains the train input data, pairs of MNIST downsampled 
            images.
        train_target: Long Tensor shape (N,)
            Contains the train input target. 1 if digit in position 0
            is less than or equal than the digit in position 2. 0 otherwise.
        train_classes: Long Tensor shape (N,2)
            Contains the classes of the digit pairs.
        n_epochs: int
            Number of epochs to train the model for.
        batch_size: int
            Number of samples per batch of training data.
        device: str
            String for device to use. Either 'cpu' or 'cuda'.
        validation_fraction: float
            Fraction of the N input samples to use for validation during
            training. Number of validation samples is computed as 
            int(N*validation_fraction).
        learning_rate: float
            Learning rate to train the model with.
        use_aux_loss: bool
            Whether to train the model 'model' using an auxiliary loss.

        Returns:
        
        all_train_losses: list len n_epochs
            List of train loss per batch.
        all_validation_losses: list len n_epochs
            List of validation loss per batch.
        all_train_acc: list len n_epochs
            List of train sorting accuracy per batch.
        all_validation_acc: list len n_epochs
            List of validation sorting accuracy per batch.
        

        """


        
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
                model.train()
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
                model.eval()
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

                    print(f"Epoch {epoch:d}. Train loss = {train_loss:.3e}. Val. loss = {val_loss:.3e}")
                    print(f"\t  Train acc = {train_accuracy:.3e}. Val. acc = {val_accuracy:.3e}")

        return all_train_losses, all_validation_losses, all_train_acc, all_validation_acc




     