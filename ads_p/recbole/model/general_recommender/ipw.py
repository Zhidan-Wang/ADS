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
from recbole.model.loss import BPRLoss, IPWLoss
from recbole.utils import InputType


class IPW(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.

    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(IPW, self).__init__(config, dataset)
        self.split_mode = config['split_mode']

        # load parameters info
        self.embedding_size = config['embedding_size']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = IPWLoss()

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
        if self.split_mode == 1:
            i_pop = interaction['pop'].numpy()
            pop = np.clip(i_pop, 1, i_pop.max() + 1)
            pop = pop/np.linalg.norm(pop, ord=np.inf)
            pop = 1/pop
            #print("pop:", pop)
            pop = np.clip(pop, 1, np.median(pop))
            pop = pop / np.linalg.norm(pop, ord=np.inf)
            pop = np.clip(pop, 0.7, 1)
            #print("pop after:", pop)
            weight = torch.tensor(pop)

        if self.split_mode == 4:
            genres = interaction['new_genre']
            # labels = interaction['rate'].to(torch.float32)
            genre_pop = []
            for u, g in zip(user, genres):
                g_pop_list = []
                user_g = user_genre[u]
                for gg in g:
                    g_pop_list.append(user_g[gg])
                g_pop = max(g_pop_list)
                genre_pop.append(g_pop)
            # print("genre_pop!!!!!", genre_pop)

            pop = np.clip(genre_pop, 0, 1)
            # pop = pop / np.linalg.norm(pop, ord=np.inf)
            pop = 1 / pop
            # print("pop:", pop)
            pop = np.clip(pop, 1, np.median(pop))
            pop = pop / np.linalg.norm(pop, ord=np.inf)
            # print("pop after:", pop)
            weight = torch.tensor(pop)
            # print("pop:", weight)

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        loss = self.loss(pos_item_score, neg_item_score, weight)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
