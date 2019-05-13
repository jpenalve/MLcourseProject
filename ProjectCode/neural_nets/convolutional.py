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
    
# +++++++++++++++++++++++ EEG NETS START  +++++++++++++++++++++++++++++++++++++++++++

class EEGNet(nn.Module): # https://arxiv.org/abs/1611.08024
    
    def __init__(self, output_dimension, dropout_perc=0.25):
        super(EEGNet, self).__init__()
        self.T = 817
        self.dropout_perc = dropout_perc
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
        x = F.dropout(x, self.dropout_perc)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, self.dropout_perc)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, self.dropout_perc)
        x = self.pooling3(x)

        # FC Layer
        x = x.view(-1,408)
        x = torch.sigmoid(self.fc1(x))
        return x


class EEGNetDeeper(nn.Module):  # https://arxiv.org/abs/1611.08024

    def __init__(self, output_dimension, dropout_perc=0.25):
        super(EEGNetDeeper, self).__init__()
        self.T = 817
        self.dropout_perc = dropout_perc
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

        # Layer 4 TIM ADDED 4 to 6
        self.padding3 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv4 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm4 = nn.BatchNorm2d(4, False)
        self.pooling4 = nn.MaxPool2d((2, 4))
        # Layer 5
        self.padding4 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv5 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm5 = nn.BatchNorm2d(4, False)
        self.pooling5 = nn.MaxPool2d((2, 4))
        # Layer 6
        self.padding5 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv6 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm6 = nn.BatchNorm2d(4, False)
        self.pooling6 = nn.MaxPool2d((2, 4))
        # L 7
        self.padding6 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv7 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm7 = nn.BatchNorm2d(4, False)
        self.pooling7 = nn.MaxPool2d((2, 4))

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
        x = F.dropout(x, self.dropout_perc)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, self.dropout_perc)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, self.dropout_perc)
        x = self.pooling3(x)

        # Layer 4
        x = self.padding3(x)
        x = F.elu(self.conv4(x))
        x = self.batchnorm4(x)
        x = F.dropout(x, self.dropout_perc)
        #x = self.pooling4(x)

        # Layer 5
        x = self.padding4(x)
        x = F.elu(self.conv5(x))
        x = self.batchnorm5(x)
        x = F.dropout(x, self.dropout_perc)
        #x = self.pooling5(x)


        # Layer 6
        x = self.padding5(x)
        x = F.elu(self.conv6(x))
        x = self.batchnorm6(x)
        x = F.dropout(x, self.dropout_perc)
        #x = self.pooling6(x)

        # Layer 7
        x = self.padding6(x)
        x = F.elu(self.conv7(x))
        x = self.batchnorm7(x)
        x = F.dropout(x, self.dropout_perc)
        #x = self.pooling7(x)

        # FC Layer
        x_flat = x.view(-1, 408)
        out = torch.sigmoid(self.fc1(x_flat))
        return out
# +++++++++++++++++++++++ EEG NETS END +++++++++++++++++++++++++++++++++++++++++++

"""class ConvNet3D(nn.Module):
    
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
    
"""

class ConvNetOzhan(nn.Module):

    def __init__(self, output_dimension):
        super(ConvNetOzhan, self).__init__()
        # Define layers
        self.convlayer = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=256, kernel_size=50, stride=1),
                                    nn.MaxPool1d(kernel_size=10, stride=10),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(0.1),
                                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=50, stride=1),
                                    nn.MaxPool1d(kernel_size=10, stride=10),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(0.1))


        self.fclayers = nn.Sequential(nn.Linear(512, 512),
                                      nn.LeakyReLU(0.1),
                                      nn.Dropout(0.5),
                                      nn.Linear(512, 256),
                                      nn.LeakyReLU(0.1),
                                      nn.Dropout(0.5),
                                      nn.Linear(256, output_dimension))

    def forward(self, x):
        x = self.convlayer(x)
        x = x.view(x.size(0), -1)
        return self.fclayers(x)