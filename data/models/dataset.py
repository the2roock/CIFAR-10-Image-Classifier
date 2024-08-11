from torch.utils.data import Dataset


class Cifar10(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.transform(self.data[idx])
        y = self.labels[idx]
        return x, y
