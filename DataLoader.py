import torch
from torch.utils.data import DataLoader, random_split
from DataSet import Custom_Data_Set

class Data_Module():
    def __init__(self, val_split: float = 0.1, batch_size: int = 32, shuffle: bool = False, seed: int = 42):
        self.val_split = val_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
    
    def data_loaders(self, train_data_set: Custom_Data_Set, val_data_set: Custom_Data_Set):
        train_dl = DataLoader(dataset = train_data_set, batch_size = self.batch_size, shuffle= self.shuffle)
        val_dl = DataLoader(dataset = val_data_set, batch_size = self.batch_size, shuffle= self.shuffle)
        return train_dl, val_dl




