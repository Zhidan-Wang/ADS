import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import itertools
import math

class MF(nn.Module):
    """
    Base module for matrix factorization.
    """
    def __init__(self, n_user, n_item, dim=64, dropout=0, init = None):
        super().__init__()
        
        self.user_latent = nn.Embedding(n_user, dim)
        self.item_latent = nn.Embedding(n_item, dim)
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=self.dropout_p)
        if init is not None:
            self.init_embedding(init)
        else: 
            self.init_embedding(0)
        
    def init_embedding(self, init): 
        
        nn.init.kaiming_normal_(self.user_latent.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_latent.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a = init)
          
    def forward(self, users, items):

        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)

        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True) + u_bias + i_bias
        # preds = u_bias + i_bias

        return preds.squeeze(dim=-1)

    def l2_norm(self, users, items): 
        users = torch.unique(users)
        items = torch.unique(items)
        
        l2_loss = (torch.sum(self.user_latent(users)**2) + torch.sum(self.item_latent(items)**2)) / 2
        return l2_loss


class OneLinear(nn.Module):
    """
    linear model: r
    """
    def __init__(self, n):
        super().__init__()
        
        self.data_bias= nn.Embedding(n, 1)
        self.init_embedding()
        
    def init_embedding(self): 
        self.data_bias.weight.data *= 0.001

    def forward(self, values):
        d_bias = self.data_bias(values)
        return d_bias.squeeze()


class TwoLinear(nn.Module):
    """
    linear model: u + i + r / o
    """
    def __init__(self, n_user, n_item):
        super().__init__()
        
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.init_embedding(0)
        
    def init_embedding(self, init): 
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a = init)

    def forward(self, users, items):

        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)
        preds = u_bias + i_bias
        return preds.squeeze()

class ThreeLinear(nn.Module):
    """
    linear model: u + i + r / o
    """
    def __init__(self, n_user, n_item, n):
        super().__init__()
        
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.data_bias= nn.Embedding(n, 1)
        self.init_embedding(0)
        
    def init_embedding(self, init): 
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.data_bias.weight, mode='fan_out', a = init)
        self.data_bias.weight.data *= 0.001

    def forward(self, users, items, values):

        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)
        d_bias = self.data_bias(values)

        preds = u_bias + i_bias + d_bias
        return preds.squeeze()

class FourLinear(nn.Module):
    """
    linear model: u + i + r + p
    """
    def __init__(self, n_user, n_item, n, n_position):
        super().__init__()
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.data_bias= nn.Embedding(n, 1)
        self.position_bias = nn.Embedding(n_position, 1)
        self.init_embedding(0)
        
    def init_embedding(self, init): 
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.data_bias.weight, mode='fan_out', a = init)
        self.data_bias.weight.data *= 0.001
        self.position_bias.weight.data *= 0.001

    def forward(self, users, items, values, positions):

        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)
        d_bias = self.data_bias(values)
        p_bias = self.position_bias(positions)

        preds = u_bias + i_bias + d_bias + p_bias
        return preds.squeeze()

class Position(nn.Module): 
    """
    the position parameters for DLA
    """
    def __init__(self, n_position): 
        super().__init__()
        self.position_bias = nn.Embedding(n_position, 1)

    def forward(self, positions): 
        return self.position_bias(positions).squeeze(dim=-1)

    def l2_norm(self, positions): 
        positions = torch.unique(positions)
        return torch.sum(self.position_bias(positions)**2)

class Discriminator_pop(nn.Module):
    def __init__(self, embedding_size, n_pop):
        super().__init__()
        self.emb_size = embedding_size
        self.hidden_size = embedding_size
        self.p_class = n_pop

        self.linear = nn.Linear(self.emb_size, self.hidden_size)
        self.act = nn.Tanh()
        self.classifer = nn.Linear(self.hidden_size, self.p_class)

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, item_emb):
        output = self.linear(item_emb)
        output = self.act(output)
        out = self.classifer(output)
        return out

    def calculate_loss(self, item_emb, p_label):
        loss = 0
        out = self.forward(item_emb)
        pred = out
        loss += self.loss_fct(pred, p_label)
        return loss

class Discriminator(nn.Module):
    def __init__(self, embedding_size, n_position):
        super().__init__()
        self.emb_size = embedding_size
        self.hidden_size = embedding_size
        self.p_class = n_position

        self.linear1 = nn.Linear(self.emb_size, self.hidden_size)
        self.linear2 = nn.Linear(self.emb_size, self.hidden_size)
        self.act = nn.Tanh()
        self.classifer1 = nn.Linear(self.hidden_size, self.p_class)
        self.classifer2 = nn.Linear(self.hidden_size, self.p_class)

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, user_emb, item_emb):
        output1 = self.linear1(item_emb)
        output1 = self.act(output1)
        out1 = self.classifer1(output1)
        if user_emb is not None:
            output2 = self.linear2(user_emb)
            output2 = self.act(output2)
            out2 = self.classifer2(output2)

            out = out1 * out2
        else:
            out = out1
        return out

    def calculate_loss(self, item_emb, user_emb, p_label):
        loss = 0
        out = self.forward(item_emb, user_emb)
        #pred = torch.zeros([out.size()[0], out.size()[1] + 1])
        #for i, o in enumerate(out):
        #    new_o = torch.cat([torch.tensor([0]), o], 0)
        #    pred[i] = new_o
        pred = out
        loss += self.loss_fct(pred, p_label)
        return loss

class MF_heckman(nn.Module): 
    def __init__(self, n_user, n_item, dim=40, dropout=0, init = None):
        super().__init__()
        self.MF = MF(n_user, n_item, dim)
        self.sigma = nn.Parameter(torch.randn(1))
    
    def forward(self, users, items, lams): 
        pred_MF = self.MF(users, items)
        pred = pred_MF - 1 * lams
        return pred

    def l2_norm(self, users, items): 
        l2_loss_MF = self.MF.l2_norm(users, items)
        l2_loss = l2_loss_MF + 1000 * torch.sum(self.sigma**2)
        return l2_loss