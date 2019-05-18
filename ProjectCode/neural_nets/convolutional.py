import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

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

    def __init__(self, output_dimension, dropout_perc=0.25):
        super(ConvNetOzhan, self).__init__()
        
        self.dropout_perc = dropout_perc
        self.k1size = 3
        self.k1stride = 3
        self.k1pool = 2
        self.k1w = 512
        
        self.k2size = 4
        self.k2stride = 2
        self.k2w = 512
        
        self.k3size = 4
        self.k3stride = 2
        self.k3w = 256
        self.k3pool = 2
        
        self.lin = self.k3w*16*8
        
        # Layer 1
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=self.k1w, kernel_size=(1,self.k1size), stride=(1,self.k1stride), padding = (0,3) ),
                                    nn.CELU(),
                                    nn.MaxPool2d(kernel_size=(1,self.k1pool), stride=(1,self.k1pool)),
                                    nn.BatchNorm2d(self.k1w,False))
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=self.k1w, out_channels=self.k2w, kernel_size=(1,self.k2size), stride=(1,self.k2stride), padding = (0,1) ),
                                    nn.CELU(),
                                    nn.BatchNorm2d(self.k2w,False))
                                   
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=self.k2w, out_channels=self.k3w, kernel_size=(self.k3size,1), stride=(self.k3stride,1), padding = (2,0) ),
                                    nn.CELU(),
                                    nn.MaxPool2d(kernel_size=(self.k3pool,1), stride=(self.k3pool,1)),
                                    nn.BatchNorm2d(self.k3w,False))

        self.fc1 =  nn.Sequential(nn.Dropout(0.25),
                                    nn.Linear(self.lin, output_dimension))
                                    

    def forward(self, x):
        x = x.view(x.size(0), 1, 64, -1)
        
        # Layer 1
        #print("L0:",x.shape)
        x = self.conv1(x)
        #print("L1:",x.shape)
        x = self.conv2(x)
        #print("L2:",x.shape)
        x = self.conv3(x)
        #print("L3:",x.shape)

        x = x.view(-1,self.k3w*16*8)
        x = torch.sigmoid(self.fc1(x))
        return x
    
    
    
    
class ConvNetOzhan3D(nn.Module):

    def __init__(self, input_dimension, output_dimension, dropout_perc=0.50):
        super(ConvNetOzhan3D, self).__init__()
        
        self.dropout_perc = dropout_perc
        self.timePoints = int(input_dimension/64)
        
        self.l1 = [ 64, (1,1,3), (1,1,3), (0,0,3), (1,1,2), (1,1,2)]
        self.l2 = [ 128, (1,1,4), (1,1,2), (0,0,1), (1,1,2), (1,1,1)]
        self.l3 = [ 128, (4,4,1), (4,4,2), (2,0,0), (2,2,1), (2,2,1)]
        
        self.dim0 = [1,8,8,self.timePoints]
        self.dim1 = self.dimout(self.dim0,self.l1)
        self.dim2 = self.dimout(self.dim1,self.l2)
        self.dim3 = self.dimout(self.dim2,self.l3)
        self.lin = int( self.dim3[0]*self.dim3[1]*self.dim3[2]*self.dim3[3] )
        
        '''
        print("\nExpected network layer shapes:")
        print(self.dim0)
        print(self.dim1)
        print(self.dim2)
        print(self.dim3)
        print(self.lin,"\n")'''

        self.conv1 = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=self.l1[0], kernel_size=self.l1[1], stride=self.l1[2], padding = self.l1[3] ),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=self.l1[4], stride=self.l1[5]),
                                    nn.BatchNorm3d(self.l1[0],False))
        self.conv2 = nn.Sequential(nn.Conv3d(in_channels=self.l1[0], out_channels=self.l2[0], kernel_size=self.l2[1], stride=self.l2[2], padding = self.l2[3] ),
                                    nn.ReLU(),
                                    #nn.MaxPool3d(kernel_size=self.l2[4], stride=self.l2[5]),
                                    nn.BatchNorm3d(self.l2[0],False))
        self.conv3 = nn.Sequential(nn.Conv3d(in_channels=self.l2[0], out_channels=self.l3[0], kernel_size=self.l3[1], stride=self.l3[2], padding = self.l3[3] ),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=self.l3[4], stride=self.l3[5]),
                                    nn.BatchNorm3d(self.l3[0],False))
        
        

        self.fc1 =  nn.Sequential(nn.Dropout(dropout_perc),
                                    nn.Linear(self.lin, output_dimension))
                                    

    def forward(self, x):
        x = x.view(x.size(0), 1, 8, 8, -1)
        #print("---Inp:",x.shape,flush=True)
        
        x = self.conv1(x)
        #print("-OutL1:",x.shape,flush=True)
        
        x = self.conv2(x)
        #print("-OutL2:",x.shape,flush=True)
        
        x = self.conv3(x)
        #print("-OutL3:",x.shape,flush=True)
        
        x = x.view(-1,self.lin)
        #print("oo-InpLin:",x.shape,"\n",flush=True)
        
        x = torch.sigmoid(self.fc1(x))
        return x
    
    
    def dimout(self, inp, l):
        out = [1,1,1,1]
        
        out[0] = l[0]
        for i in range(3):
            out[i+1] = np.floor( ( inp[i+1] + 2*l[3][i] - (l[1][i] - 1) - 1 ) / l[2][i] + 1 ) 
            out[i+1] = int( out[i+1] / l[5][i] )
        return out
        