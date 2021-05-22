#!/usr/bin/env python
import torch

class ConvNet(torch.nn.Module):
    """
    Module to implement simple LeNet type convnet.

    Attributes:

    in_channels: int
        Number of channels in the input.
    out_channels_1: int
        Number of output channels of first convolutional layer.
    out_channels_2: int
        Number of output channels of second convolutional layer.
    kernel_size_1: int
        Kernel size of first convolutional layer.
    kernel_size_2: int
        Kernel size of second convolutional layer.
    n_hidden: int
        Number of hidden layers in the Linear layer.
    n_classes: int
        Number of output units (classes).
    conv: Model
        Convolutional part of the model.
    dense: Model
        Fully connected part of the model.

    Methods:

    forward:
        Compute the output of the model.

    """


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
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(self.in_channels, self.out_channels_1, self.kernel_size_1, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2),
                                        torch.nn.Conv2d(self.out_channels_1, self.out_channels_2, self.kernel_size_2, stride=1, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size=2))
        dummy_out = self.conv(self.dummy)
        # n_elem = 1
        # for d in dummy_out.shape:
        #   n_elem = n_elem * d #(if the kernel size is different then 3 you might get a buggy answer, what do you think?)
        n_elem = dummy_out.shape[-1] * dummy_out.shape[-2] * dummy_out.shape[-3]
        self.dense = torch.nn.Sequential(torch.nn.Linear(in_features = n_elem, out_features = self.n_hidden),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features = self.n_hidden, out_features = self.n_classes))

    def forward(self, x):
        """
        Compute the output of the model

        Parameters

        x: Tensor shape (batch_size, n_channels, 14, 14)
            Input to the model
        
        Returns

        out: Tensor shape (batch_size, n_classes)
            Predictions of the model

        """

        out = self.conv(x)
        out = self.dense(out.view(out.shape[0], -1))
        return out


class NN(torch.nn.Module):

    """
    Module to implement simple LeNet type convnet.

    Attributes:
    
    out_channels_1: int
        out_channels_1 to pass to ConvNet.
    out_channels_2: int
        out_channels_2 to pass to ConvNet.
    kernel_size_1: int
        kernel_size_1 to pass to ConvNet.
    kernel_size_2: int
        kernel_size_2 to pass to ConvNet.
    n_hidden: int
        n_hidden to pass to ConvNet.
    conv1: Model
        A ConvNet to classify the first digit.
    conv2: Model
        A ConvNet to classify the second digit.
    dense: Model
        MLP to predict sorting from digit predictions.

    Methods:

    forward:
        Compute the outputs of the model (digit classes and sorting).

    """

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
        """
        Compute the output of the model

        Parameters

        input: Tensor shape (batch_size, n_channels, 14, 14)
            Input to the model
        
        Returns

        out_1: Tensor shape (batch_size, 10)
            First digit classification predictions. Class prediction 
            can be computed as out_1.argmax(axis=1).
        out_2: Tensor shape (batch_size, 10)
            Second digit classification predictions. Class prediction 
            can be computed as out_2.argmax(axis=1).
        out: Tensor shape (batch_size, 2)
            Sorting predictions. Class prediction can be computed as
            out.argmax(axis=1).

        """

        in_1 = input[:, 0, :, :].unsqueeze(1)
        in_2 = input[:, 1, :, :].unsqueeze(1)
        out_1 = self.conv1(in_1)
        out_2 = self.conv2(in_2)
        out = self.dense(torch.cat((out_1, out_2), dim=1))
        return out_1, out_2, out

class NN_ws(torch.nn.Module):

    """
    Module to implement simple LeNet type convnet.

    Attributes:
    
    out_channels_1: int
        out_channels_1 to pass to ConvNet.
    out_channels_2: int
        out_channels_2 to pass to ConvNet.
    kernel_size_1: int
        kernel_size_1 to pass to ConvNet.
    kernel_size_2: int
        kernel_size_2 to pass to ConvNet.
    n_hidden: int
        n_hidden to pass to ConvNet.
    conv: Model
        A ConvNet to classify the digits.
    dense: Model
        MLP to predict sorting from digit predictions.

    Methods:

    forward:
        Compute the outputs of the model (digit classes and sorting).

    """

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
        """
        Compute the output of the model

        Parameters

        input: Tensor shape (batch_size, n_channels, 14, 14)
            Input to the model
        
        Returns

        out_1: Tensor shape (batch_size, 10)
            First digit classification predictions. Class prediction 
            can be computed as out_1.argmax(axis=1).
        out_2: Tensor shape (batch_size, 10)
            Second digit classification predictions. Class prediction 
            can be computed as out_2.argmax(axis=1).
        out: Tensor shape (batch_size, 2)
            Sorting predictions. Class prediction can be computed as
            out.argmax(axis=1).

        """

        in_1 = input[:, 0, :, :].unsqueeze(1)
        in_2 = input[:, 1, :, :].unsqueeze(1)
        out_1 = self.conv(in_1)
        out_2 = self.conv(in_2)
        out = self.dense(torch.cat((out_1, out_2), dim=1))
        return out_1, out_2, out


class Baseline(torch.nn.Module):

    """
    Module to implement simple LeNet type convnet.

    Attributes:
    
    out_channels_1: int
        out_channels_1 to pass to ConvNet.
    out_channels_2: int
        out_channels_2 to pass to ConvNet.
    kernel_size_1: int
        kernel_size_1 to pass to ConvNet.
    kernel_size_2: int
        kernel_size_2 to pass to ConvNet.
    n_hidden: int
        n_hidden to pass to ConvNet.
    conv: Model
        A ConvNet to pass the input data through.
    dense: Model
        MLP to predict sorting from conv's output.

    Methods:

    forward:
        Compute the outputs of the model (sorting).

    """

    def __init__(self, out_channels_1=32, out_channels_2=64, kernel_size_1=5, kernel_size_2=3, n_hidden=200):
        super().__init__()

        self.out_channels_1 = out_channels_1
        self.out_channels_2 = out_channels_2
        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2
        self.n_hidden = n_hidden

        # Define layers for network
        
        # A ConvNet to identify the digits
        self.conv = ConvNet(in_channels=2, n_classes=32, 
                            out_channels_1=self.out_channels_1,
                            out_channels_2=self.out_channels_2,
                            kernel_size_1=self.kernel_size_1,
                            kernel_size_2=self.kernel_size_2,
                            n_hidden=self.n_hidden)

        # A dense network to know which is larger
        self.dense = torch.nn.Sequential(#torch.nn.Linear(in_features = 20, out_features = 32),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features = 32, out_features = 64),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features = 64, out_features = 2),
                                        torch.nn.Softmax(dim=1))



    def forward(self, input):
        """
        Compute the output of the model

        Parameters

        input: Tensor shape (batch_size, n_channels, 14, 14)
            Input to the model
        
        Returns

        out: Tensor shape (batch_size, 2)
            Sorting predictions. Class prediction can be computed as
            out.argmax(axis=1).

        """
        
        out = self.conv(input)
        out = self.dense(out)
        return out


