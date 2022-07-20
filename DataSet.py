import torch
from torch.utils.data import Dataset

class Custom_Data_Set(Dataset):
    def __init__(self, data):
        self.x = torch.tensor(data[0:len(data)-1], dtype = torch.float)
        self.y = torch.tensor(data[1:], dtype = torch.float)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]














        # print(len(data)-1)
        # print(self.x.shape)
        # print(self.y.shape)
        # print(self.y[0])
        # print(self.x[0])
        # print(self.x[1])