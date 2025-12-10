# This class is a user-defined Root Mean Square Error Loss function, built upon pytorch MSE function
import torch.nn as nn
import torch  

mse =nn.MSELoss()

class rmse(nn.Module):
    def forward(self, preds, golds):
        return torch.sqrt(mse(preds, golds))
