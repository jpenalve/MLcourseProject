import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F







class ConvNet2DFFT(nn.Module): # https://mediatum.ub.tum.de/doc/1422453/552605125571.pdf

    def __init__(self, output_dimension):
        super(ConvNet2DFFT, self).__init__()
        # Define layers
        
        
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=24, kernel_size=12, stride=1, padding=5),
                                    nn.BatchNorm2d(24),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.ReLU(),
                                    nn.Dropout(0.5))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=24, out_channels=48, kernel_size=8, stride=1, padding=(4,3)),
                                    nn.BatchNorm2d(48),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.ReLU(),
                                    nn.Dropout(0.5))
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=48, out_channels=96, kernel_size=4, stride=1, padding=(2,2)),
                                    nn.BatchNorm2d(96),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.ReLU(),
                                    nn.Dropout(0.5))
        

        self.fclayers = nn.Sequential(nn.Linear(6144, output_dimension))

    def forward(self, input):
        bsize = input.shape[0]
        ch = input.shape[1]
        T = input.shape[2]
        h_length=12
        n = 128
        
        output = input.view(bsize*ch,-1)
        output = torch.stft(output,n,hop_length=h_length,normalized=True)
        output = output[:,:,:,0]
        output = output.view(bsize,ch,int(n/2+1),int(T/h_length+1))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = output.view(bsize,-1)
        output = self.fclayers(output)
        return output
        