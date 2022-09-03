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
import numpy as np

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class BPR_PC(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.

    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BPR_PC, self).__init__(config, dataset)

        # load parameters info
        self.alpha = config['alpha'] or 0.5
        self.beta = config['beta'] or 0.6
        self.embedding_size = config['embedding_size']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        r""" Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        r""" Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding(item)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, interaction):
        #print("interraction!!!!!!!!", interaction)
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        loss = self.loss(pos_item_score, neg_item_score)
        return loss

    def predict(self, interaction):
        print("predict!!!!!!!!")
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction, item_popularity):
        #print("sort_predict!!!!!!!!!!!")
        #print(interaction)
        user = interaction[self.USER_ID]

        used_list = interaction['used_list']
        un_used_num = interaction['un_used_num']
        un_used_num = un_used_num.unsqueeze(1)

        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        #print("score:", score)

        U_n = score * used_list
        U_n /= un_used_num
        U_n = torch.norm(U_n, dim=-1)
        #print("U_n:", U_n.size())

        C_u = score * self.beta + (1 - self.beta)
        item_popularity = torch.tensor(item_popularity)
        item_pop = item_popularity.expand_as(score)
        #pop = np.clip(item_pop, 1, item_pop.max() + 1)
        #pop = pop / np.linalg.norm(pop, ord=np.inf)
        #pop = 1 / pop
        # print("pop:", pop)
        #pop = np.clip(pop, 1, np.median(pop))
        #pop = pop / np.linalg.norm(pop, ord=np.inf)

        #weight = torch.tensor(pop)
        #C_u *= weight
        C_u /= item_pop
        #print("C_u:", C_u)

        U_c = C_u * used_list
        U_c /= un_used_num
        U_c = U_c[:, 1:]
        U_c = torch.norm(U_c, dim=-1)
        #print("U_c:", U_c)

        scale = U_n / U_c
        scale = scale.reshape(-1, 1)
        compensate = C_u * scale
        #print("score:", score)
        #print("compensate:", compensate)
        score += self.alpha * compensate

        return score.view(-1)
