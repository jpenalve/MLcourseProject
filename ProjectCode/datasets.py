# Here we define the different dataset classes we possibly need for different approaches
from torch.utils.data import Dataset


# We do no transformation, just transforming to tensor
class FlatLabelsDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index]
        if self.transform:
            data = self.transform(data)
        labels = self.target[index]
        return data, labels

    def __len__(self):
        return len(self.data)
