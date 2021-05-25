#!/usr/bin/env python
import torch

class ResBlock(torch.nn.Module):
    """
    Module describing a single ResNet block.

    Attributes

    conv1: Module
        First convolutional layer of the block.
    bn1: Module
        First batch normalization layer of the block.
    conv2: Module
        Second convolutional layer of the block.
    bn2: Module
        Second batch normalization layer of the block.

    Methods

    forward:
        Compute the output of the block

    """

    def __init__(self, n_channels, kernel_size):
        """
        Parameters

        n_channels: int
            Number of input channels
        kernel_size: int
            Kernel size of convolutional layers. This defines the padding
            as padding=(kernel_size - 1) // 2

        """


        super().__init__()
        self.conv1 = torch.nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=(kernel_size -1 ) // 2)
        self.bn1 = torch.nn.BatchNorm2d(n_channels)
        self.conv2 = torch.nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=(kernel_size -1 ) // 2)
        self.bn2 = torch.nn.BatchNorm2d(n_channels)
       
    def forward(self, x):
        """
        Compute the output of the block

        Parameters

        x: Tensor shape (batch_size, n_channels, 14, 14)

        Returns

        y: Tensor shape (batch_size, n_channels, 14, 14)

        """

        y = self.bn1(self.conv1(x))
        y = torch.nn.functional.relu(y)
        y = self.bn2(self.conv2(y))
        y += x
        y = torch.nn.functional.relu(y)
        return y

class ResNet(torch.nn.Module):
    """
    Module to implement ResNet.

    Attributes:
 
    conv0: Module
        Convolutional layer to cast input into Tensor
        of shape (batch_size, n_channels, 14, 14)
    resblocks: Module
        Sequential model composed of n_blocks ResBlocks.
    avg: Module
        Average pool to reduce Tensor to shape 
        (batch_size, n_channels, 1, 1)
    fc: Module
        Fully connected layer to get out_channels predictions.

    Methods:

    forward:
        Compute the output of the model.

    """
    def __init__(self, n_channels, kernel_size, n_blocks, in_channels=1,out_channels=10):
        """
        Parameters

        n_channels: int
            Number of input channels
        kernel_size: int
            Kernel size of convolutional layers. This defines the padding
            as padding=(kernel_size - 1) // 2
        n_blocks: int
            Number of ResBlocks to use in ResNet
        in_channels: int
            Number of input channels in the data
        out_channels: int
            Number of output channels (predictions)

        """
        super().__init__()
        self.conv0 = torch.nn.Conv2d(in_channels, n_channels, kernel_size=1)
        self.resblocks = torch.nn.Sequential( *(ResBlock(n_channels, kernel_size) for _ in range(n_blocks)))
        self.avg = torch.nn.AvgPool2d(kernel_size=14)
        self.fc = torch.nn.Linear(n_channels, out_channels)

    def forward(self, x):
        """
        Compute the output of the block

        Parameters

        x: Tensor shape (batch_size, n_channels, 14, 14)

        Returns

        x: Tensor shape (batch_size, out_channels)

        """
        x = torch.nn.functional.relu(self.conv0(x))
        x = self.resblocks(x)
        x = torch.nn.functional.relu(self.avg(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class ResNet_NN(torch.nn.Module):
    """
    Module to implement ResNet for digit pair sorting.

    Attributes:
    
    resnet1: Module
        ResNet to classify the first digit.
    resnet2: Module
        ResNet to classify the second digit.
    dense: Module
        MLP to get sorting prediction from digit predictions

    Methods:

    forward:
        Compute the outputs of the model (digit classes and sorting).

    """

    def __init__(self, n_channels=16, kernel_size=3, n_blocks=25):
        """
        Parameters

        n_channels: int
            Number of input channels
        kernel_size: int
            Kernel size of convolutional layers. This defines the padding
            as padding=(kernel_size - 1) // 2
        n_blocks: int
            Number of ResBlocks to use in ResNet

        """
        super().__init__()

        self.resnet1 = ResNet(n_channels, kernel_size, n_blocks)
        self.resnet2 = ResNet(n_channels, kernel_size, n_blocks)

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
        out_1 = self.resnet1(in_1)
        out_2 = self.resnet2(in_2)
        out = self.dense(torch.cat((out_1, out_2), dim=1))
        return out_1, out_2, out

class ResNet_NN_ws(torch.nn.Module):
    """
    Module to implement ResNet for digit pair sorting
    with weight sharing.

    Attributes:
    
    resnet: Module
        ResNet to classify both digits.
    dense: Module
        MLP to get sorting prediction from digit predictions

    Methods:

    forward:
        Compute the outputs of the model (digit classes and sorting).

    """

    def __init__(self, n_channels=16, kernel_size=3, n_blocks=25):
        """
        Parameters

        n_channels: int
            Number of input channels
        kernel_size: int
            Kernel size of convolutional layers. This defines the padding
            as padding=(kernel_size - 1) // 2
        n_blocks: int
            Number of ResBlocks to use in ResNet
        
        """
        super().__init__()

        self.resnet = ResNet(n_channels, kernel_size, n_blocks)

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
        out_1 = self.resnet(in_1)
        out_2 = self.resnet(in_2)
        out = self.dense(torch.cat((out_1, out_2), dim=1))
        return out_1, out_2, out

class ResNet_Baseline(torch.nn.Module):

    """
    Module to implement ResNet for digit pair sorting
    without separating digits.

    Attributes:
    
    resnet: Module
        ResNet
    dense: Module
        MLP to get sorting prediction.

    Methods:

    forward:
        Compute the outputs of the model (digit classes and sorting).

    """
    
    def __init__(self, n_channels=16, kernel_size=3, n_blocks=25, in_channels=2):
        """
        Parameters

        n_channels: int
            Number of input channels
        kernel_size: int
            Kernel size of convolutional layers. This defines the padding
            as padding=(kernel_size - 1) // 2
        n_blocks: int
            Number of ResBlocks to use in ResNet
        in_channels: int
            Number of input channels in the data

        """
        super().__init__()

        self.resnet = ResNet(n_channels, kernel_size, n_blocks, in_channels=in_channels,
                            out_channels = 32)

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

        input: Tensor shape (batch_size, in_channels, 14, 14)
            Input to the model
        
        Returns

        out: Tensor shape (batch_size, 2)
            Sorting predictions. Class prediction can be computed as
            out.argmax(axis=1).

        """

        out = self.resnet(input)
        out = self.dense(out)
        return out
