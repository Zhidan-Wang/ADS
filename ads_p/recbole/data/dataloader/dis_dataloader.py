
import numpy as np, torch
from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader
from recbole.data.dataloader.neg_sample_mixin import NegSampleByMixin, NegSampleMixin
from recbole.data.interaction import Interaction, cat_interactions
from recbole.utils import DataLoaderType, FeatureSource, FeatureType, InputType

class DisDataLoader(AbstractDataLoader):
    dl_type = DataLoaderType.ORIGIN

    def __init__(self, config, dataset, batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.time_field = dataset.time_field

        self.max_item_list_len = config['MAX_ITEM_LIST_LENGTH']
        list_suffix = config['LIST_SUFFIX']

        for field in dataset.inter_feat:
            if field != self.uid_field:
                list_field = field + list_suffix
                setattr(self, f"{field}_list_field", list_field)
                ftype = dataset.field2type[field]
                if ftype in (FeatureType.TOKEN, FeatureType.TOKEN_SEQ):
                    list_ftype = FeatureType.TOKEN_SEQ
                else:
                    list_ftype = FeatureType.FLOAT_SEQ
                if ftype in (FeatureType.TOKEN_SEQ, FeatureType.FLOAT_SEQ):
                    list_len = (
                     self.max_item_list_len, dataset.field2seqlen[field])
                else:
                    list_len = self.max_item_list_len

                dataset.set_field_property(list_field, list_ftype, FeatureSource.INTERACTION, list_len)

        self.item_list_length_field = config['ITEM_LIST_LENGTH_FIELD']
        self.label_field = config['LABEL_FIELD']

        dataset.set_field_property(self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1)
        dataset.set_field_property(self.label_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1)

        self.uid_list = dataset.uid_list
        self.item_list_index = dataset.item_list_index
        self.target_index = dataset.target_index
        self.item_list_length = dataset.item_list_length
        self.label = dataset.label
        self.pre_processed_data = None

        super().__init__(config, dataset, batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

    def data_preprocess(self):
        """Do data augmentation before training/evaluation.
        """
        self.pre_processed_data = self.augmentation(self.item_list_index, self.target_index, self.item_list_length, self.label)

    @property
    def pr_end(self):
        return len(self.uid_list)

    def _shuffle(self):
        if self.real_time:
            new_index = torch.randperm(self.pr_end)
            self.uid_list = self.uid_list[new_index]
            self.item_list_index = self.item_list_index[new_index]
            self.target_index = self.target_index[new_index]
            self.label = self.label[new_index]
            self.item_list_length = self.item_list_length[new_index]
        else:
            self.pre_processed_data.shuffle()

    def _next_batch_data(self):
        cur_data = self._get_processed_data(slice(self.pr, self.pr + self.step))
        self.pr += self.step
        return cur_data

    def _get_processed_data(self, index):
        if self.real_time:
            cur_data = self.augmentation(self.item_list_index[index], self.target_index[index], self.item_list_length[index], self.label[index])
        else:
            cur_data = self.pre_processed_data[index]
        return cur_data

    def augmentation(self, item_list_index, target_index, item_list_length, label):
        """Data augmentation.

        Args:
            item_list_index (numpy.ndarray): the index of history items list in interaction.
            target_index (numpy.ndarray): the index of items to be predicted in interaction.
            item_list_length (numpy.ndarray): history list length.

        Returns:
            dict: the augmented data.
        """
        new_length = len(item_list_index)
        new_data = self.dataset.inter_feat[target_index]

        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
            self.label_field: torch.tensor(label)
        }

        for field in self.dataset.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f"{field}_list_field")
                list_len = self.dataset.field2seqlen[list_field]
                shape = (new_length, list_len) if isinstance(list_len, int) else (new_length,) + list_len
                list_ftype = self.dataset.field2type[list_field]
                dtype = torch.int64 if list_ftype in (FeatureType.TOKEN, FeatureType.TOKEN_SEQ) else torch.float64
                new_dict[list_field] = torch.zeros(shape, dtype=dtype)

                value = self.dataset.inter_feat[field]
                for i, (index, length) in enumerate(zip(item_list_index, item_list_length)):
                    new_dict[list_field][i][:length] = value[index]

        new_data.update(Interaction(new_dict))
        #print("before shuffle label!!!!!!", new_data['label'])
        #print('new_data', new_data['item_id_list'])
        #print("new_data", new_data['user_id'])
        return new_data