from torch.utils.data import Dataset


class SupervisedDatasets(Dataset):

    def __init__(self, data, labels, transform=None):
        self.transform = transform

        self.data = data
        self.label = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """sample = {'eda': self.data[idx, :, 0], 'bvp': self.data[idx, :, 1],
                  'temp': self.data[idx, :, 2], 'label': self.label[idx]}

        if self.transform:
            sample = self.transform(sample)"""

        return self.data[idx], self.label[idx]

