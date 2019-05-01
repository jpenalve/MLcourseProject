import torch.nn as nn
import torch.nn.functional as F
import torch


class ConvNet01(nn.Module):

    def __init__(self, output_dimension):
        super(ConvNet01, self).__init__()
        # Define layers
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=128, kernel_size=5, stride=1),
                                    nn.LeakyReLU(0.2),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1),
                                    nn.LeakyReLU(0.2),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.fclayers = nn.Sequential(nn.Linear(334464, 384),
                                      nn.LeakyReLU(0.2),
                                      nn.Dropout(0.5),
                                      nn.Linear(384, 192),
                                      nn.LeakyReLU(0.2),
                                      nn.Dropout(0.5),
                                      nn.Linear(192, output_dimension))

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = output.view(output.size(0), -1)
        output = self.fclayers(output)
        return output
    
    
class ConvNet1D(nn.Module):

    def __init__(self, output_dimension):
        super(ConvNet1D, self).__init__()
        # Define layers
        self.convlayer = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1),
                                    nn.LeakyReLU(0.2),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1),
                                    nn.LeakyReLU(0.2),
                                    nn.MaxPool2d(kernel_size=2, stride=2))


        self.fclayers = nn.Sequential(nn.Linear(6432, 384),
                                      nn.LeakyReLU(0.2),
                                      nn.Dropout(0.5),
                                      nn.Linear(384, 192),
                                      nn.LeakyReLU(0.2),
                                      nn.Dropout(0.5),
                                      nn.Linear(192, output_dimension))

    def forward(self, x):
        x = self.convlayer(x)
        x = x.view(x.size(0), -1)
        return self.fclayers(x)
    
    
class EEGNet(nn.Module):
    
    def __init__(self, output_dimension):
        super(EEGNet, self).__init__()
        self.T = 817

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.fc1 = nn.Linear(408, output_dimension)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.view(x.size(0), 1, -1, 64)
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # FC Layer
        x = x.view(-1,408)
        x = torch.sigmoid(self.fc1(x))
        return x
    
    
    
    
class ConvNet3D(nn.Module):
    
    # Will be done 3D
    def __init__(self, output_dimension):
        super(ConvNet3D, self).__init__()
        # Define layers
        self.convlayer = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1),
                                    nn.LeakyReLU(0.2),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1),
                                    nn.LeakyReLU(0.2),
                                    nn.MaxPool2d(kernel_size=2, stride=2))


        self.fclayers = nn.Sequential(nn.Linear(6432, 384),
                                      nn.LeakyReLU(0.2),
                                      nn.Dropout(0.5),
                                      nn.Linear(384, 192),
                                      nn.LeakyReLU(0.2),
                                      nn.Dropout(0.5),
                                      nn.Linear(192, output_dimension))

    def forward(self, x):
        x = self.convlayer(x)
        x = x.view(x.size(0), -1)
        return self.fclayers(x)
    
    
    
    
    
    
    
    
    
# PyTorch implementation of EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces
# (Vernon, 2016)
# TODO: This net has to be adapted to our shape
"""
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.fc1 = nn.Linear(4 * 2 * 7, 1)

    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # FC Layer
        x = x.view(-1, 4 * 2 * 7)
        x = F.sigmoid(self.fc1(x))
        return x
"""