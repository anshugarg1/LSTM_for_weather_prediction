import torch
from Model import Weather_Model
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class train_model():
    def __init__(self, criterion, model, device, writer, optimiser: optim, epochs: int):
        self.criterion = criterion
        self.model = model
        self.optimiser = optimiser
        self.epochs = epochs
        self.device = device
        self.writer = writer

    def train_func(self, train_loader: DataLoader, val_loader: DataLoader):
        for epoch in tqdm(range(self.epochs)):
            train_loss_ls = []
            val_loss_ls = []
            total_train_loss = 0.0
            total_val_loss = 0.0

            for batch_data in train_loader:
                x,y = batch_data
                x = torch.reshape(x, (x.shape[0], 1, x.shape[1])).to(self.device)  #[32, 1, 6]
                y = torch.reshape(y, (y.shape[0], 1, y.shape[1])).to(self.device)  #[32, 1, 6]

                self.optimiser.zero_grad()
                pred,_ = self.model(x)  #[32, 1, 6]
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimiser.step()
                total_train_loss += loss.item()

            loss = total_train_loss/len(train_loader)   
            train_loss_ls.append(loss)  
            self.writer.add_scalar('Loss/train', loss, epoch)
            if epoch%9 == 0:
                print("Training Loss {} at epoch {}".format(loss, epoch))

            for val_batch in val_loader:
                val_x, val_y = val_batch
                val_x = torch.reshape(val_x, (val_x.shape[0], 1, val_x.shape[1])).to(self.device)  #[32, 1, 6]
                val_y = torch.reshape(val_y, (val_y.shape[0], 1, val_y.shape[1])).to(self.device)  #[32, 1, 6]

                val_pred,_ = self.model(val_x)  #[32, 1, 6]
                val_loss = self.criterion(val_pred, val_y)
                total_val_loss += val_loss

            val_loss = total_val_loss/len(val_loader)
            val_loss_ls.append(val_loss)
            self.writer.add_scalar('Loss/validation', val_loss, epoch)
            if epoch%9 == 0:
                print("Validation Loss at epoch {} is {}".format(val_loss.item(), epoch))

