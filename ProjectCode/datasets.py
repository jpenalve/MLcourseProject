# Here we define the different dataset classes we possibly need for different approaches
from torch.utils.data import Dataset
import torch

# We do no transformation, just transforming to tensor
class ChannelsVoltageDataset(Dataset):
    def __init__(self, data, target, normalize, transform=None):
        self.data = data
        self.target = target
        self.transform = transform
        self.normalize = normalize

    def __getitem__(self, index):
        data = self.data[index]
        
        #if self.normalize:
        #    mean = torch.mean(data)
        #    std = torch.std(data)
        #    data = (data - mean) / std
        #    data = (data+1)*0.5
        
        if self.transform:
            data = self.transform(data)
        
        labels = self.target[index]
        return data, labels

    def __len__(self):
        return len(self.data)
