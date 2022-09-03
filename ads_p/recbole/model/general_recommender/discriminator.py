import torch
import torch.nn as nn
import numpy as np

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class Discriminator(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(Discriminator, self).__init__(config, dataset)
        self.emb_size = config['embedding_size']
        self.hidden_size = config['embedding_size']
        #self.p_class = config['p_class']
        print("self.emb:", self.emb_size)
        print("self.hidden:", self.hidden_size)
        self.linear = nn.Linear(self.emb_size, self.hidden_size)
        self.act = nn.Tanh()
        self.classifer = nn.Linear(self.hidden_size, 2)

        self.loss_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, item_emb):
        output = self.linear(item_emb)
        output = self.act(output)
        out = self.classifer(output)
        return out

    def calculate_loss(self, item_emb, p_label):
        pred = self.forward(item_emb)
        loss = self.loss_fct(pred, p_label)
        return loss

    def predict(self, item_emb, p_label):
        out = self.forward(item_emb)
        preds = np.argmax(out, axis=1)
        n_hits = (p_label == preds).nonzero(as_tuple=False)[:, :-1].size(0)

        scores = n_hits
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)