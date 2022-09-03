# -*- coding: utf-8 -*-
# @Time   : 2020/6/27
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/22,
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmain.com


import torch
import torch.nn as nn
from torch.nn.init import normal_

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.layers import MLPLayers
from recbole.utils import InputType


class MF(GeneralRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(MF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.dropout_prob = config['dropout_prob']
        self.weight_decay = config['weight_decay']
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.item_embedding1 = nn.Embedding(self.n_items, self.embedding_size)
        #self.user_bias = nn.Embedding(self.n_users, 1)
        #self.item_bias = nn.Embedding(self.n_items, 1)

        self.dropout = nn.Dropout(p=self.dropout_prob)

        # parameters initialization
        self.init_embedding(0)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss(reduction='sum')

    def init_embedding(self, init):

        nn.init.kaiming_normal_(self.user_embedding.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.item_embedding.weight, mode='fan_out', a=init)
        #nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a=init)
        #nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a=init)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user, item):
        u_latent = self.dropout(self.user_embedding(user))
        i_latent = self.dropout(self.item_embedding(item))
        #u_bias = self.user_bias(user)
        #i_bias = self.item_bias(item)
        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True)
        out = preds.squeeze(dim=-1)
        #print("out:", out)
        #preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True) + u_bias + i_bias
        #preds = u_bias + i_bias
        return out

    def forward1(self, user, item):
        u_latent = self.dropout(self.user_embedding(user))
        i_latent = self.dropout(self.item_embedding1(item))
        #u_bias = self.user_bias(user)
        #i_bias = self.item_bias(item)
        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True)
        out = preds.squeeze(dim=-1)
        #print("out:", out)
        #preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True) + u_bias + i_bias
        #preds = u_bias + i_bias
        return out

    def l2_norm(self, user, item):
        users = torch.unique(user)
        items = torch.unique(item)

        l2_loss = (torch.sum(self.user_embedding(users) ** 2) + torch.sum(self.item_embedding(items) ** 2)) / 2
        return l2_loss

    def l2_norm1(self, user, item):
        users = torch.unique(user)
        items = torch.unique(item)

        l2_loss = (torch.sum(self.user_embedding(users) ** 2) + torch.sum(self.item_embedding1(items) ** 2)) / 2
        return l2_loss

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        #label = interaction['rate'].to(torch.float32)
        label = interaction['label']
        output = self.forward(user, item)

        loss = self.loss(output, label) + self.weight_decay * self.l2_norm(user, item)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

    def dump_parameters(self):
        r"""A simple implementation of dumping model parameters for pretrain.

        """
        if self.mf_train and not self.mlp_train:
            save_path = self.mf_pretrain_path
            torch.save(self, save_path)
        elif self.mlp_train and not self.mf_train:
            save_path = self.mlp_pretrain_path
            torch.save(self, save_path)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
