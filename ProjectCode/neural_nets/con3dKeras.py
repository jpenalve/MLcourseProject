import torch.nn as nn
import torch.nn.functional as F
import torch

#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Flatten
#from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D
#from keras.optimizers import SGD

import keras as KR
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten 
from keras.layers import Conv3D, MaxPooling3D, ZeroPadding3D 
from keras.optimizers import SGD 
from kapre.time_frequency import Spectrogram  
from time_frequence import stft
from stft import STFT

class ConvNet3DKeras(nn.Module): ## https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2 ; https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf

    def __init__(self, output_dimension):
        super(ConvNet3DKeras, self).__init__()
        # Define layers
        self.layer1 = SequentialK(Conv3D(64, 3, 3, 3, activation='relu', border_mode='same', name='conv1', subsample=(1, 1, 1), input_shape=(3, 16, 112, 112)),MaxPooling3D(pool_size=(1,2, 2), strides=(1,2, 2), border_mode='valid', name='pool1'))
        #layer2
        self.layer2 = SequentialK(KR.Conv3DK(128, 3, 3, 3, activation='relu',border_mode='same', name='conv2', subsample=(1, 1, 1)),MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool2'))
        
        #layer3
        self.layer3 = SequentialK(KR.Conv3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3a',subsample=(1, 1, 1)),Conv3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3b',subsample=(1, 1, 1)),MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool3'))
        
        #layer4
        self.layer4 = SequentialK(Conv3D(512, 3, 3, 3, activation='relu',border_mode='same', name='conv4a',subsample=(1, 1, 1)),Conv3D(512, 3, 3, 3, activation='relu',border_mode='same', name='conv4b',subsample=(1, 1, 1)),MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),border_mode='valid', name='pool4'))
        
        #layer5
        self.layer5 = SequentialK(Conv3D(512, 3, 3, 3, activation='relu',border_mode='same', name='conv5a',subsample=(1, 1, 1)),Conv3D(512, 3, 3, 3, activation='relu',border_mode='same', name='conv5b',subsample=(1, 1, 1)),ZeroPadding3D(padding=(0, 1, 1)),MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool5'),Flatten())
        
        
        # FC layers 
        self.fclayers = SequentialK(Dense(4096, activation='relu', name='fc6'),Dense(.5),Dense(4096, activation='relu', name='fc7'),Dense(.5),Dense(487, activation='softmax', name='fc8'))

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        output = output.view(output.size(0), -1)
        output = self.fclayers(output)
        return output
        

class ConvNet3DFFT(nn.Module): # https://mediatum.ub.tum.de/doc/1422453/552605125571.pdf

    def __init__(self, output_dimension):
        super(ConvNet3DFFT, self).__init__()
        # Define layers
        
        self.layer1 = STFT() # https://github.com/keunwoochoi/kapre/blob/master/kapre/time_frequency.py 
        # Spectrogram https://pytorch.org/audio/_modules/torchaudio/transforms.html USE THIS WITH NUMBER OF JUMPS ... 
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=24, kernel_size=12, stride=1),
                                    nn.BatchNorm2d(24, False),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.ReLU(),
                                    nn.Dropout(0.5))
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=24, out_channels=48, kernel_size=8, stride=1),
                                    nn.BatchNorm2d(48, False),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.ReLU(),
                                    nn.Dropout(0.5))
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=48, out_channels=96, kernel_size=4, stride=1),
                                    nn.BatchNorm2d(96, False),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.ReLU(),
                                    nn.Dropout(0.5))
        

        self.fclayers = nn.Sequential(nn.Softmax())

    def forward(self, input):
        print(input.shape)

        output = input.view(input.shape[0], 1, -1)
        print(output.shape)

        output = self.layer1(output)  
        print(output.shape)
        
        #print(output.shape)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = output.view(output.size(0), -1)
        output = self.fclayers(output)
        return output
        