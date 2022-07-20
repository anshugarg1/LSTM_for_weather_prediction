import torch
import torch.nn as nn
import torch.nn.functional as F

class Weather_Model(nn.Module):
    def __init__(self, device, inp_dim:int = 6, hid_dim1:int = 128, hid_dim2: int = 50, out_dim: int = 6, 
    num_layers: int = 2, bias: bool = False, batch_first : bool = True):
        super(Weather_Model, self).__init__()
        self.device = device
        self.inp_dim = inp_dim
        self.hid_dim1 = hid_dim1
        self.hid_dim2 = hid_dim2
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size = self.inp_dim, hidden_size = self.hid_dim1, num_layers = self.num_layers)
        self.lstm2 = nn.LSTM(input_size = self.hid_dim1, hidden_size = self.hid_dim2, num_layers = self.num_layers)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(1), self.hid_dim1).to(self.device) #hidden state
        c_0 = torch.zeros(self.num_layers, x.size(1), self.hid_dim1).to(self.device) #cell state

        h_2 = torch.zeros(self.num_layers, x.size(1), self.hid_dim2).to(self.device) #hidden state
        c_2 = torch.zeros(self.num_layers, x.size(1), self.hid_dim2).to(self.device) #cell state

        out, (h, c) = self.lstm(x, (h_0, c_0)) # out shape- [32, 1, 6], c.shape- [2, 1, 6], h.shape- [2, 1, 6]
        out2, (h2, c2) = self.lstm2(out, (h_2, c_2)) # out shape- [32, 1, 6], c.shape- [2, 1, 6], h.shape- [2, 1, 6]
        # print("out2 shape", out2.shape)
        return out2, (h2,c2)
