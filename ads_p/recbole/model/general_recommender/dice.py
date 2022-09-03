# -*- coding: utf-8 -*-
# @Time   : 2020/6/25
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/16
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
BPR
################################################
Reference:
    Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.
"""

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
import math
import numpy as np


class DICE(GeneralRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(DICE, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size =int(config['embedding_size'] / 2)
        print("emb_size:", self.embedding_size)
        print("user:", self.n_users)

        # define layers and loss
        self.user_int_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_int_embedding = nn.Embedding(self.n_items, self.embedding_size)

        self.user_pop_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_pop_embedding = nn.Embedding(self.n_items, self.embedding_size)

        self.alpha = config['alpha']
        self.beta = config['beta']
        print("alpha:", self.alpha)
        print("beta:", self.beta)

        self.loss = BPRLoss()
        self.criterion_discrepancy = self.dcor

        # parameters initialization
        #self.apply(xavier_normal_initialization)
        self.init_params()

    def init_params(self):
        stdv = 1. / math.sqrt(self.user_int_embedding.weight.size(1))
        self.user_int_embedding.weight.data.uniform_(-stdv, stdv)
        self.user_pop_embedding.weight.data.uniform_(-stdv, stdv)
        self.item_int_embedding.weight.data.uniform_(-stdv, stdv)
        self.item_pop_embedding.weight.data.uniform_(-stdv, stdv)

    def dcor(self, x, y):
        a = torch.norm(x[:, None] - x, p=2, dim=2)
        b = torch.norm(y[:, None] - y, p=2, dim=2)

        A = a - a.mean(dim=0)[None, :] - a.mean(dim=1)[:, None] + a.mean()
        B = b - b.mean(dim=0)[None, :] - b.mean(dim=1)[:, None] + b.mean()

        n = x.size(0)

        dcov2_xy = (A * B).sum() / float(n * n)
        dcov2_xx = (A * A).sum() / float(n * n)
        dcov2_yy = (B * B).sum() / float(n * n)
        dcor = -torch.sqrt(dcov2_xy) / torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

        return dcor

    def get_user_embedding(self, user):
        r""" Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        user_pop = self.user_pop_embedding(user)
        user_int = self.user_int_embedding(user)
        user_e = torch.cat((user_int, user_pop), 1)
        return user_e

    def get_item_embedding(self, item):
        r""" Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        item_pop = self.item_pop_embedding(item)
        item_int = self.item_int_embedding(item)
        item_e = torch.cat((item_int, item_pop), 1)
        return item_e

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        pos_item_pop = interaction['pop']
        neg_item_pop = interaction['neg_pop']

        #print("pos_pop:", pos_pop)
        #print("neg_pop:", neg_pop)

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_score, neg_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        rank_loss = self.loss(pos_score, neg_score)
        #print("rank_loss:", rank_loss)

        user_int = self.user_int_embedding(user)
        pos_int = self.item_int_embedding(pos_item)
        neg_int = self.item_int_embedding(neg_item)
        pos_int_score, neg_int_score = torch.mul(user_int, pos_int).sum(dim=1), torch.mul(user_int, neg_int).sum(dim=1)

        user_pop = self.user_pop_embedding(user)
        pos_pop = self.item_pop_embedding(pos_item)
        neg_pop = self.item_pop_embedding(neg_item)
        pos_pop_score, neg_pop_score = torch.mul(user_pop, pos_pop).sum(dim=1), torch.mul(user_pop, neg_pop).sum(dim=1)

        #print("pos_int_score:", pos_int_score)

        mask = pos_item_pop <= neg_item_pop
        #print("mask:", mask)
        pos_int_score1 = pos_int_score[mask]
        #print("after:", pos_int_score)
        neg_int_score1 = neg_int_score[mask]

        int_loss = self.loss(pos_int_score1, neg_int_score1)
        #print("int_loss:", int_loss)

        pos_pop_score1 = pos_pop_score[mask]
        neg_pop_score1 = neg_pop_score[mask]

        pop_loss = self.loss(neg_pop_score1, pos_pop_score1)

        mask2 = pos_item_pop > neg_item_pop
        pos_pop_score2 = pos_pop_score[mask2]
        neg_pop_score2 = neg_pop_score[mask2]

        pop_loss += self.loss(pos_pop_score2, neg_pop_score2)
        #print("pop_loss:", pop_loss)

        item_all = torch.unique(torch.cat((pos_item, neg_item)))
        item_int = self.item_int_embedding(item_all)
        item_pop = self.item_pop_embedding(item_all)
        user_all = torch.unique(user)
        user_int = self.user_int_embedding(user_all)
        user_pop = self.user_pop_embedding(user_all)
        discrepency_loss = self.criterion_discrepancy(item_int, item_pop) + self.criterion_discrepancy(user_int,user_pop)
        #print("dis_loss:", discrepency_loss)

        loss = rank_loss + int_loss + self.alpha * pop_loss + self.beta * discrepency_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = torch.cat((self.item_int_embedding.weight, self.item_pop_embedding.weight), 1)
        #user_e = self.user_int_embedding(user)
        #all_item_e = self.item_int_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
