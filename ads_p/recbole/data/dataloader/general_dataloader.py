# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2020/9/9, 2020/9/29
# @Author : Yupeng Hou, Yushuo Chen
# @email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn

"""
recbole.data.dataloader.general_dataloader
################################################
"""

import numpy as np
import torch

from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader
from recbole.data.dataloader.neg_sample_mixin import NegSampleMixin, NegSampleByMixin
from recbole.data.interaction import Interaction, cat_interactions
from recbole.utils import DataLoaderType, InputType


class GeneralDataLoader(AbstractDataLoader):
    """:class:`GeneralDataLoader` is used for general model and it just return the origin data.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~recbole.utils.enum_type.InputType.POINTWISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """
    dl_type = DataLoaderType.ORIGIN

    def __init__(self, config, dataset, batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        super().__init__(config, dataset, batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)
        self.uid_field = dataset.uid_field  # session_id
        self.iid_field = dataset.iid_field  # item_id
        self.config = config
        self.model_name = self.config['model']

    @property
    def pr_end(self):
        return len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        #cur_data = self.dataset[self.pr:self.pr + self.step]
        cur_data = self._augment(self.dataset[self.pr:self.pr + self.step])
        self.pr += self.step
        return cur_data

    def _augment(self, inter_feat):
        #print("*********************",inter_feat['genre'])
        pos_ids = inter_feat[self.iid_field]
        pos_pop = self._get_item_pop(pos_ids)
        #print("!!!!!!!!!!!!!!!!!", pos_pop)
        pos_pop_feat = Interaction({'pop': pos_pop})
        pos_pop_feat = self.dataset.join(pos_pop_feat)

        inter_feat.update(pos_pop_feat)

        if self.model_name == 'BPR_PC':
            user_ids = inter_feat[self.uid_field]
            used_set= self.sampler.get_used_ids()['train'][user_ids]
            #print("!!!!!!!!!!!!!!", used)
            used = []
            unused_num = []
            for u in used_set:
                l = np.full(self.dataset.item_num, 0)
                for i in list(u):
                    ###### +1????
                    l[i] = 1
                #print("lllllllllllllll:", l)
                used.append(l)
                unused_num.append(self.dataset.item_num - len(list(u)))
                #print("u:", u)
            #print("!!!!!!!!!!!!!!", unused_num)
            used_list = Interaction({'used_list': torch.tensor(used)})
            unused_num_list = Interaction({'un_used_num': torch.tensor(unused_num)})

            inter_feat.update(used_list)
            inter_feat.update(unused_num_list)

        #print("new_data_after_neg:", inter_feat)
        return inter_feat

    def _get_item_pop(self, iids):
        iids = np.array(iids)
        pop = []
        for iid in iids:
            pop.append(self.dataset.item_inter_num[iid])
        return torch.tensor(pop)


class GeneralNegSampleDataLoader(NegSampleByMixin, AbstractDataLoader):
    """:class:`GeneralNegSampleDataLoader` is a general-dataloader with negative sampling.
    For the result of every batch, we permit that every positive interaction and its negative interaction
    must be in the same batch. Beside this, when it is in the evaluation stage, and evaluator is topk-like function,
    we also permit that all the interactions corresponding to each user are in the same batch
    and positive interactions are before negative interactions.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        neg_sample_args (dict): The neg_sample_args of dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~recbole.utils.enum_type.InputType.POINTWISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """
    def __init__(
        self, config, dataset, sampler, neg_sample_args, batch_size=1, dl_format=InputType.POINTWISE, shuffle=False
    ):
        self.uid_field = dataset.uid_field #session_id
        self.iid_field = dataset.iid_field #item_id
        self.uid_list, self.uid2index, self.uid2items_num = None, None, None
        self.model = config['model']

        super().__init__(
            config, dataset, sampler, neg_sample_args, batch_size=batch_size, dl_format=dl_format, shuffle=shuffle
        ) #setup() data_preprocess()

    def setup(self):
        if self.user_inter_in_one_batch: #Flase
            uid_field = self.dataset.uid_field
            user_num = self.dataset.user_num
            self.dataset.sort(by=uid_field, ascending=True)
            self.uid_list = []
            start, end = dict(), dict()
            for i, uid in enumerate(self.dataset.inter_feat[uid_field].numpy()):
                if uid not in start:
                    self.uid_list.append(uid)
                    start[uid] = i
                end[uid] = i
            self.uid2index = np.array([None] * user_num)
            self.uid2items_num = np.zeros(user_num, dtype=np.int64)
            for uid in self.uid_list:
                self.uid2index[uid] = slice(start[uid], end[uid] + 1)
                self.uid2items_num[uid] = end[uid] - start[uid] + 1
            self.uid_list = np.array(self.uid_list)
        self._batch_size_adaptation()

    def _batch_size_adaptation(self):
        if self.user_inter_in_one_batch:
            inters_num = sorted(self.uid2items_num * self.times, reverse=True)
            batch_num = 1
            new_batch_size = inters_num[0]
            for i in range(1, len(inters_num)):
                if new_batch_size + inters_num[i] > self.batch_size:
                    break
                batch_num = i + 1
                new_batch_size += inters_num[i]
            self.step = batch_num
            self.upgrade_batch_size(new_batch_size)
        else:
            batch_num = max(self.batch_size // self.times, 1)
            new_batch_size = batch_num * self.times
            self.step = batch_num
            self.upgrade_batch_size(new_batch_size)

    @property
    def pr_end(self):
        if self.user_inter_in_one_batch:
            return len(self.uid_list)
        else:
            return len(self.dataset)

    def _shuffle(self):
        if self.user_inter_in_one_batch:
            np.random.shuffle(self.uid_list)
        else:
            self.dataset.shuffle()

    def _next_batch_data(self):
        if self.user_inter_in_one_batch:
            uid_list = self.uid_list[self.pr:self.pr + self.step]
            data_list = []
            for uid in uid_list:
                index = self.uid2index[uid]
                data_list.append(self._neg_sampling(self.dataset[index]))
            cur_data = cat_interactions(data_list)
            pos_len_list = self.uid2items_num[uid_list]
            user_len_list = pos_len_list * self.times
            cur_data.set_additional_info(list(pos_len_list), list(user_len_list))
            self.pr += self.step
            return cur_data
        else:
            cur_data = self._neg_sampling(self.dataset[self.pr:self.pr + self.step])
            self.pr += self.step
            return cur_data

    def _neg_sampling(self, inter_feat):
        uids = inter_feat[self.uid_field]

        neg_iids = self.sampler.sample_by_user_ids(uids, self.neg_sample_by)
        return self.sampling_func(inter_feat, neg_iids)

    def _neg_sample_by_pair_wise_sampling(self, inter_feat, neg_iids):
        inter_feat = inter_feat.repeat(self.times)
        #print("*************", inter_feat['genre'])
        neg_item_feat = Interaction({self.iid_field: neg_iids})
        neg_item_feat = self.dataset.join(neg_item_feat)
        neg_item_feat.add_prefix(self.neg_prefix)

        pos_ids = inter_feat[self.iid_field]
        pos_pop = self._get_item_pop(pos_ids)
        #print("!!!!!!!!!!!!!!!!!", pos_pop)
        pos_pop_feat = Interaction({'pop': pos_pop})
        pos_pop_feat = self.dataset.join(pos_pop_feat)
        #neg_item_feat.add_prefix(self.neg_prefix)

        neg_pop = self._get_item_pop(neg_iids)
        #print("!!!!!!!!!!!!!!!!!", neg_pop)
        neg_pop_feat = Interaction({'pop': neg_pop})
        neg_pop_feat = self.dataset.join(neg_pop_feat)
        neg_pop_feat.add_prefix(self.neg_prefix)

        inter_feat.update(neg_item_feat)
        inter_feat.update(pos_pop_feat)
        inter_feat.update(neg_pop_feat)

        if self.config['split_mode'] == 4:
            neg_new_g = self._reclass_genre(inter_feat)
            #print("neg_new_genre:::", neg_new_g)
            neg_new_genre = Interaction({'new_genre': neg_new_g})
            neg_new_genre = self.dataset.join(neg_new_genre)
            neg_new_genre.add_prefix(self.neg_prefix)

            inter_feat.update(neg_new_genre)

        #print("new_data_after_neg:", inter_feat)
        return inter_feat

    def _get_item_pop(self, iids):
        iids = np.array(iids)
        pop = []
        for iid in iids:
            pop.append(self.dataset.item_inter_num[iid])
        return torch.tensor(pop)

    def _get_item_genre(self, item_file):
        item_num = self.dataset.item_num
        item_mp = self.dataset.item_mp
        genre_mp = self.dataset.gener_mp
        #print("item_mp:::!!!", item_mp)
        #print("genre_mp:::****", genre_mp)
        item_genre = torch.zeros([item_num, 6], dtype=torch.long)

        with open(item_file, 'r') as f:
            lines = f.readlines()
            for id, line in enumerate(lines):
                if id > 0:
                    item_id, movie_title, year, genre = line.strip().split('\t')
                    genre_list = genre.strip().split(' ')
                    g_list = []
                    for g in genre_list:
                        gid = self.get_index(genre_mp, g)
                        g_list.append(gid)
                    while len(g_list) < 6:
                        g_list.append(0)
                    i_id = self.get_index(item_mp, item_id)
                    if i_id > 0:
                        item_genre[i_id] = torch.tensor(g_list)
        item_genre = torch.LongTensor(item_genre)
        return item_genre

    def _reclass_genre(self, inter_feat):
        g_map =[0, 1, 1, 2, 3, 4, 5, 6, 3, 7, 7, 7, 4, 8, 3, 1, 7, 6, 3]

        origin_genre_list = inter_feat['neg_genre']
        new_genre_list = torch.zeros_like(origin_genre_list)
        for gid, origin_genre in enumerate(origin_genre_list):
            new_genre = []
            for origin in origin_genre:
                if origin == 0:
                    break
                new_genre.append(g_map[origin])
            new_genre = list(set(new_genre))

            while (len(new_genre) < len(origin_genre)):
                new_genre.append(0)
            new_genre_list[gid] = torch.tensor(new_genre)
        #print("new_genre_list::::",new_genre_list)
        return new_genre_list

    def get_index(self, mp, value):
        for id, v in enumerate(mp):
            if v == value:
                return id + 1
        return -1

    def _neg_sample_by_point_wise_sampling(self, inter_feat, neg_iids):
        pos_inter_num = len(inter_feat)
        new_data = inter_feat.repeat(self.times)
        new_data[self.iid_field][pos_inter_num:] = neg_iids
        new_data = self.dataset.join(new_data)
        labels = torch.zeros(pos_inter_num * self.times)
        labels[:pos_inter_num] = 1.0
        new_data.update(Interaction({self.label_field: labels}))
        return new_data

    def get_pos_len_list(self):
        """
        Returns:
            numpy.ndarray: Number of positive item for each user in a training/evaluating epoch.
        """
        return self.uid2items_num[self.uid_list]

    def get_user_len_list(self):
        """
        Returns:
            numpy.ndarray: Number of all item for each user in a training/evaluating epoch.
        """
        return self.uid2items_num[self.uid_list] * self.times


class GeneralFullDataLoader(NegSampleMixin, AbstractDataLoader):
    """:class:`GeneralFullDataLoader` is a general-dataloader with full sort. In order to speed up calculation,
    this dataloader would only return then user part of interactions, positive items and used items.
    It would not return negative items.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        neg_sample_args (dict): The neg_sample_args of dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~recbole.utils.enum_type.InputType.POINTWISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """
    dl_type = DataLoaderType.FULL

    def __init__(
        self, config, dataset, sampler, neg_sample_args, batch_size=1, dl_format=InputType.POINTWISE, shuffle=False
    ):
        if neg_sample_args['strategy'] != 'full':
            raise ValueError('neg_sample strategy in GeneralFullDataLoader() should be `full`')

        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        user_num = dataset.user_num
        self.model_name = config['model']
        self.uid_list = []
        self.uid2positive = np.array([None] * user_num)
        self.uid2items_num = np.zeros(user_num, dtype=np.int64)
        self.uid2swap_idx = np.array([None] * user_num)
        self.uid2rev_swap_idx = np.array([None] * user_num)
        self.uid2history_item = np.array([None] * user_num)

        self.uid2swap_idx1 = np.array([None] * user_num)
        self.uid2rev_swap_idx1 = np.array([None] * user_num)
        self.uid2swap_idx2 = np.array([None] * user_num)
        self.uid2rev_swap_idx2 = np.array([None] * user_num)

        dataset.sort(by=self.uid_field, ascending=True)
        last_uid = None
        positive_item = set()
        uid2used_item = sampler.used_ids
        #print("!!!!!!!!dataset.inter_feat", dataset.inter_feat)
        for uid, iid in zip(dataset.inter_feat[self.uid_field].numpy(), dataset.inter_feat[self.iid_field].numpy()):
            if uid != last_uid:
                #if last_uid == 103:
                    #print("uid222222222:", last_uid)
                    #print("uid2used_item22222222222:", uid2used_item[last_uid])

                    #print("positive_item22222222222:", positive_item)
                self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
                last_uid = uid
                self.uid_list.append(uid)
                positive_item = set()
            positive_item.add(iid)

        self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
        self.uid_list = torch.tensor(self.uid_list)
        #self.user_df = dataset
        self.user_df = dataset.join(Interaction({self.uid_field: self.uid_list}))
        #print("self.user_df!!!!!!!!!!:", self.uid_list)


        super().__init__(
            config, dataset, sampler, neg_sample_args, batch_size=batch_size, dl_format=dl_format, shuffle=shuffle
        )

    def _set_user_property(self, uid, used_item, positive_item):
        if uid is None:
            return

        history_item = used_item - positive_item
        positive_item_num = len(positive_item)
        #print("uid:", uid)
        #print("used_item:", used_item)
        #print("history_item:", history_item)
        self.uid2positive[uid] = positive_item
        self.uid2items_num[uid] = positive_item_num
        #print("positive_item_num:", positive_item_num)
        swap_idx1 = torch.tensor(sorted(set(range(positive_item_num))))
        swap_idx2 = torch.tensor(sorted(positive_item))

        self.uid2swap_idx1[uid] = swap_idx1
        self.uid2rev_swap_idx2[uid] = swap_idx1.flip(0)

        self.uid2swap_idx2[uid] = swap_idx2
        self.uid2rev_swap_idx1[uid] = swap_idx2.flip(0)
        #print("swap_idx1:", swap_idx1)
        #print("swap_re1:", self.uid2rev_swap_idx1[uid])
        #print("swap_idx2:", swap_idx2)
        #print("swap_re2:", self.uid2rev_swap_idx2[uid])

        #print("flip:", self.uid2rev_swap_idx[uid])
        self.uid2history_item[uid] = torch.tensor(list(history_item), dtype=torch.int64)

    def _set_user_property1(self, uid, used_item, positive_item):
        if uid is None:
            return

        history_item = used_item - positive_item
        positive_item_num = len(positive_item)
        #print("uid:", uid)
        #print("used_item:", used_item)
        #print("history_item:", history_item)
        self.uid2items_num[uid] = positive_item_num
        #print("positive_item_num:", positive_item_num)
        swap_idx = torch.tensor(sorted(set(range(positive_item_num)) ^ positive_item))
        print("swap_idx!!!!!!!!!!!!!!!:", swap_idx)
        self.uid2swap_idx[uid] = swap_idx
        self.uid2rev_swap_idx[uid] = swap_idx.flip(0)
        #print("flip:", self.uid2rev_swap_idx[uid])
        self.uid2history_item[uid] = torch.tensor(list(history_item), dtype=torch.int64)

    def _batch_size_adaptation(self):
        batch_num = max(self.batch_size // self.dataset.item_num, 1)
        new_batch_size = batch_num * self.dataset.item_num
        ###################
        self.step = self.batch_size
        #self.step = batch_num
        self.upgrade_batch_size(new_batch_size)

    @property
    def pr_end(self):
        return len(self.uid_list)

    def _shuffle(self):
        self.logger.warnning('GeneralFullDataLoader can\'t shuffle')

    def _next_batch_data(self):
        if self.model_name == 'BPR_PC':
            user_df = self._augment(self.user_df[self.pr:self.pr + self.step])
        else:
            user_df = self.user_df[self.pr:self.pr + self.step]
        cur_data = self._neg_sampling(user_df)
        self.pr += self.step
        return cur_data

    def _neg_sampling(self, user_df):
        uid_list = list(user_df[self.dataset.uid_field])
        pos_len_list = self.uid2items_num[uid_list] #1
        #print("pos_len_list:!!!!!!!!", pos_len_list)
        user_len_list = np.full(len(uid_list), self.item_num) # 20612
        #print("user_len_list:!!!!!!!!", user_len_list)
        user_df.set_additional_info(pos_len_list, user_len_list)

        history_item = self.uid2history_item[uid_list]
        #print("history:", history_item)
        history_row = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_item)])
        #print("history_row:", history_row)
        history_col = torch.cat(list(history_item))
        #print("history_col:", history_col)

        swap_idx1 = self.uid2swap_idx1[uid_list]
        rev_swap_idx1 = self.uid2rev_swap_idx1[uid_list]
        swap_idx2 = self.uid2swap_idx2[uid_list]
        rev_swap_idx2 = self.uid2rev_swap_idx2[uid_list]
        #print("swap_idx:", swap_idx)
        swap_row1 = torch.cat([torch.full_like(swap, i) for i, swap in enumerate(swap_idx1)])
        swap_row2 = torch.cat([torch.full_like(swap, i) for i, swap in enumerate(swap_idx2)])
        #print("swap_row:", swap_row1)
        swap_col_after1 = torch.cat(list(swap_idx1))
        #print("swap_col_after:", swap_col_after1)
        swap_col_before1 = torch.cat(list(rev_swap_idx1))
        #print("swap_col_before:", swap_col_before1)

        swap_col_after2 = torch.cat(list(swap_idx2))
        swap_col_before2 = torch.cat(list(rev_swap_idx2))
        return user_df, (history_row, history_col), swap_row1, swap_col_after1, swap_col_before1, swap_row2, swap_col_after2, swap_col_before2

    def _neg_sampling1(self, user_df):
        uid_list = list(user_df[self.dataset.uid_field])
        pos_len_list = self.uid2items_num[uid_list] #1
        #print("pos_len_list:!!!!!!!!", pos_len_list)
        user_len_list = np.full(len(uid_list), self.item_num) # 20612
        #print("user_len_list:!!!!!!!!", user_len_list)
        user_df.set_additional_info(pos_len_list, user_len_list)

        history_item = self.uid2history_item[uid_list]
        #print("history:", history_item)
        history_row = torch.cat([torch.full_like(hist_iid, i) for i, hist_iid in enumerate(history_item)])
        #print("history_row:", history_row)
        history_col = torch.cat(list(history_item))
        #print("history_col:", history_col)

        swap_idx = self.uid2swap_idx[uid_list]
        rev_swap_idx = self.uid2rev_swap_idx[uid_list]
        #print("swap_idx:", swap_idx)
        swap_row = torch.cat([torch.full_like(swap, i) for i, swap in enumerate(swap_idx)])
        #print("swap_row:", swap_row)
        swap_col_after = torch.cat(list(swap_idx))
        #print("swap_col_after:", swap_col_after)
        swap_col_before = torch.cat(list(rev_swap_idx))
        #print("swap_col_before:", swap_col_before)
        return user_df, (history_row, history_col), swap_row, swap_col_after, swap_col_before

    def _augment(self, inter_feat):
        user_ids = inter_feat[self.uid_field]
        used_set= self.sampler.get_used_ids()['train'][user_ids]
        #print("!!!!!!!!!!!!!!", used)
        used = []
        unused_num = []
        for u in used_set:
            l = np.full(self.dataset.item_num, 1)
            for i in list(u):
                ###### +1????
                l[i] = 0
            #print("lllllllllllllll:", l)
            used.append(l)
            unused_num.append(self.dataset.item_num - len(list(u)))
            #print("u:", u)
        #print("!!!!!!!!!!!!!!", unused_num)
        used_list = Interaction({'used_list': torch.tensor(used)})
        unused_num_list = Interaction({'un_used_num': torch.tensor(unused_num)})

        inter_feat.update(used_list)
        inter_feat.update(unused_num_list)

        #print("new_data_after_neg:", inter_feat)
        return inter_feat

    def get_pos_len_list(self):
        """
        Returns:
            numpy.ndarray: Number of positive item for each user in a training/evaluating epoch.
        """
        return self.uid2items_num[self.uid_list]

    def get_user_len_list(self):
        """
        Returns:
            numpy.ndarray: Number of all item for each user in a training/evaluating epoch.
        """
        return np.full(self.pr_end, self.item_num)
