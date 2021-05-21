#!/usr/bin/env python
import torch

class ResBlock(torch.nn.Module):
    def __init__(self, nb_channels, kernel_size):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(nb_channels, nb_channels, kernel_size=kernel_size, padding=(kernel_size -1 ) // 2)
        self.bn1 = torch.nn.BatchNorm2d(nb_channels)
        self.conv2 = torch.nn.Conv2d(nb_channels, nb_channels, kernel_size=kernel_size, padding=(kernel_size -1 ) // 2)
        self.bn2 = torch.nn.BatchNorm2d(nb_channels)
       
    def forward(self, x):
        y = self.bn1(self.conv1(x))
        y = torch.nn.functional.relu(y)
        y = self.bn2(self.conv2(y))
        y += x
        y = torch.nn.functional.relu(y)
        return y

class ResNet(torch.nn.Module):
    def __init__(self, nb_channels, kernel_size, nb_blocks, in_channels=1,out_channels=10):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(in_channels, nb_channels, kernel_size=1)
        self.resblocks = torch.nn.Sequential( *(ResBlock(nb_channels, kernel_size) for _ in range(nb_blocks)))
        self.avg = torch.nn.AvgPool2d(kernel_size=14)
        self.fc = torch.nn.Linear(nb_channels, out_channels)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv0(x))
        x = self.resblocks(x)
        x = torch.nn.functional.relu(self.avg(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class ResNet_NN(torch.nn.Module):
    def __init__(self, nb_channels=16, kernel_size=3, nb_blocks=25):
        super().__init__()

        self.resnet1 = ResNet(nb_channels, kernel_size, nb_blocks)
        self.resnet2 = ResNet(nb_channels, kernel_size, nb_blocks)

        self.dense = torch.nn.Sequential(torch.nn.Linear(in_features = 20, out_features = 32),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features = 32, out_features = 64),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features = 64, out_features = 2),
                                        torch.nn.Softmax(dim=1))

    def forward(self, input):
        in_1 = input[:, 0, :, :].unsqueeze(1)
        in_2 = input[:, 1, :, :].unsqueeze(1)
        out_1 = self.resnet1(in_1)
        out_2 = self.resnet2(in_2)
        out = self.dense(torch.cat((out_1, out_2), dim=1))
        return out_1, out_2, out

class ResNet_NN_ws(torch.nn.Module):
    def __init__(self, nb_channels=16, kernel_size=3, nb_blocks=25):
        super().__init__()

        self.resnet = ResNet(nb_channels, kernel_size, nb_blocks)

        self.dense = torch.nn.Sequential(torch.nn.Linear(in_features = 20, out_features = 32),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features = 32, out_features = 64),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features = 64, out_features = 2),
                                        torch.nn.Softmax(dim=1))

    def forward(self, input):
        in_1 = input[:, 0, :, :].unsqueeze(1)
        in_2 = input[:, 1, :, :].unsqueeze(1)
        out_1 = self.resnet(in_1)
        out_2 = self.resnet(in_2)
        out = self.dense(torch.cat((out_1, out_2), dim=1))
        return out_1, out_2, out

class ResNet_Baseline(torch.nn.Module):
    def __init__(self, nb_channels=16, kernel_size=3, nb_blocks=25, in_channels=2):
        super().__init__()

        self.resnet = ResNet(nb_channels, kernel_size, nb_blocks, in_channels=in_channels,
                            out_channels = 32)

        self.dense = torch.nn.Sequential(#torch.nn.Linear(in_features = 20, out_features = 32),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features = 32, out_features = 64),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(in_features = 64, out_features = 2),
                                        torch.nn.Softmax(dim=1))

    def forward(self, input):
        out = self.resnet(input)
        out = self.dense(out)
        return out
