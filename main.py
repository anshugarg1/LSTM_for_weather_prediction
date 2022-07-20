import torch 
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from DataPreprocessing import Data_Preprocess 
from DataSet import Custom_Data_Set
from DataLoader import Data_Module
from Model import Weather_Model
from train import train_model

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

if __name__ == '__main__':
    #constants 
    epochs = 1000
    lr = 1e-3
    file_path = '../../../varshneya/WeatherData/NPZ-HOH-atomar-2018.txt'
    plot_graph = False
    val_split = 30
    batch_size = 32, 
    shuffle = True, 
    seed = 42
    inp_dim = 6
    hid_dim1 = 128 #6 
    hid_dim2 = 6
    out_dim = 6 
    num_layers = 1 #2
    bias = False
    batch_first = True

    device = get_device()
    writer = SummaryWriter()
    dp = Data_Preprocess(file_path, plot_graph)
    data = dp.read_file()
    
    train_ds = Custom_Data_Set(data[:len(data)-val_split]) 
    val_ds = Custom_Data_Set(data[len(data)-val_split:])
    
    dl = Data_Module(val_split = 0.1, batch_size = 32, shuffle = True, seed = 42)
    train_dl, val_dl = dl.data_loaders(train_ds, val_ds)

    model = Weather_Model(device, inp_dim, hid_dim1, hid_dim2, out_dim, num_layers, bias, batch_first).to(device)

    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr)
    tm = train_model(criterion, model, device, writer, optimiser, epochs)
    tm.train_func(train_dl, val_dl)
    writer.close()
    # print(data)
    