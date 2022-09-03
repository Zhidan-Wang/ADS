# @Time   : 2020/6/28
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2020/10/28 2020/10/13, 2020/11/10
# @Author : Yupeng Hou, Xingyu Pan, Yushuo Chen
# @Email  : houyupeng@ruc.edu.cn, panxy@ruc.edu.cn, chenyushuo@ruc.edu.cn

"""
recbole.data.dataset
##########################
"""

import copy
import json
import os
from collections import Counter
from logging import getLogger

import numpy as np
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn_utils
from scipy.sparse import coo_matrix

from recbole.data.interaction import Interaction, cat_interactions
from recbole.data.utils import dlapi
from recbole.utils import FeatureSource, FeatureType


class Dataset(object):
    """:class:`Dataset` stores the original dataset in memory.
    It provides many useful functions for data preprocessing, such as k-core data filtering and missing value
    imputation. Features are stored as :class:`pandas.DataFrame` inside :class:`~recbole.data.dataset.dataset.Dataset`.
    General and Context-aware Models can use this class.

    By calling method :meth:`~recbole.data.dataset.dataset.Dataset.build()`, it will processing dataset into
    DataLoaders, according to :class:`~recbole.config.eval_setting.EvalSetting`.

    Args:
        config (Config): Global configuration object.
        saved_dataset (str, optional): Restore Dataset object from ``saved_dataset``. Defaults to ``None``.

    Attributes:
        dataset_name (str): Name of this dataset.

        dataset_path (str): Local file path of this dataset.

        field2type (dict): Dict mapping feature name (str) to its type (:class:`~recbole.utils.enum_type.FeatureType`).

        field2source (dict): Dict mapping feature name (str) to its source
            (:class:`~recbole.utils.enum_type.FeatureSource`).
            Specially, if feature is loaded from Arg ``additional_feat_suffix``, its source has type str,
            which is the suffix of its local file (also the suffix written in Arg ``additional_feat_suffix``).

        field2id_token (dict): Dict mapping feature name (str) to a :class:`np.ndarray`, which stores the original token
            of this feature. For example, if ``test`` is token-like feature, ``token_a`` is remapped to 1, ``token_b``
            is remapped to 2. Then ``field2id_token['test'] = ['[PAD]', 'token_a', 'token_b']``. (Note that 0 is
            always PADDING for token-like features.)

        field2token_id (dict): Dict mapping feature name (str) to a dict, which stores the token remap table
            of this feature. For example, if ``test`` is token-like feature, ``token_a`` is remapped to 1, ``token_b``
            is remapped to 2. Then ``field2token_id['test'] = {'[PAD]': 0, 'token_a': 1, 'token_b': 2}``.
            (Note that 0 is always PADDING for token-like features.)

        field2seqlen (dict): Dict mapping feature name (str) to its sequence length (int).
            For sequence features, their length can be either set in config,
            or set to the max sequence length of this feature.
            For token and float features, their length is 1.

        uid_field (str or None): The same as ``config['USER_ID_FIELD']``.

        iid_field (str or None): The same as ``config['ITEM_ID_FIELD']``.

        label_field (str or None): The same as ``config['LABEL_FIELD']``.

        time_field (str or None): The same as ``config['TIME_FIELD']``.

        inter_feat (:class:`Interaction`): Internal data structure stores the interaction features.
            It's loaded from file ``.inter``.

        user_feat (:class:`Interaction` or None): Internal data structure stores the user features.
            It's loaded from file ``.user`` if existed.

        item_feat (:class:`Interaction` or None): Internal data structure stores the item features.
            It's loaded from file ``.item`` if existed.

        feat_name_list (list): A list contains all the features' name (:class:`str`), including additional features.
    """

    def __init__(self, config, gene_batch=None, saved_dataset=None):
        self.config = config
        self.dataset_name = config['dataset']
        self.logger = getLogger()
        self._dataloader_apis = {'field2type', 'field2source', 'field2id_token'}
        self._dataloader_apis.update(dlapi.dataloader_apis)
        self.item_popularity = None

        if saved_dataset is None:
            self._from_scratch()
            self.user_genre = np.zeros((self.user_num, 9))
        else:
            self._restore_saved_dataset(saved_dataset)

        if config['add_generator'] == True:
            print("***************************************************")
            print("before:", self.inter_feat)
            if config['if_gan'] == True:
                self.g_file = config['generator_path'] + str(gene_batch) + 'gene.data'
                self.g_inter_path = config['generator_path'] + str(gene_batch) + 'gene.inter'
            else:
                #self.g_file = 'dataset/generator/narm_pretrain/' + 'gene.data'
                #self.g_inter_path = 'dataset/generator/narm_pretrain/' + 'gene.inter'

                self.g_file = config['generator_path'] + 'gene.data'
                self.g_inter_path = config['generator_path'] + 'gene.inter'
            print("gene data path:", self.g_file)

            #gene_df = self._load_feat(self.g_inter_path, FeatureSource.INTERACTION)
            #gene = self._dataframe_to_interaction(gene_df)

            if config['dataset'] == 'ml-1m':
                add_num = 200000
            else:
                add_num = 50000
            gene = self.get_gene_inter(self.g_file, self.g_inter_path, add_num, config['dataset'])
            #print("gene length", len(gene))
            self.inter_feat = cat_interactions([self.inter_feat, gene])
            print("after:", self.inter_feat)

    def get_gene_inter(self, data_file, inter_file, add_num, dataset):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        with open(inter_file, 'w') as fn:
            user_num = self.user_num
            if dataset == 'diginetica':
                fn.write('session_id:token' + '\t' + 'item_id:token' + '\t' + 'timestamp:float' + '\n')
            else:
                fn.write('user_id:token' + '\t' + 'item_id:token' + '\t' + 'timestamp:float' + '\n')
            for i, line in enumerate(lines):
                l = line.strip().split(' ')
                star = np.random.randint(0, 2)
                for j, s in enumerate(l):
                    if j == star:
                        continue
                    else:
                        fn.write(str(user_num + i) + '\t' + str(s) + '\t' + str(j) + '\n')

                #x = np.random.randint(1, self.item_num)
                #fn.write(str(user_num + i) + '\t' + str(x) + '\t' + str(j) + '\n')


        gene_df = self._load_feat(inter_file, FeatureSource.INTERACTION)
        gene_df = gene_df[:add_num]

        gene = self._dataframe_to_interaction(gene_df)
        return gene

    def _from_scratch(self):
        """Load dataset from scratch.
        Initialize attributes firstly, then load data from atomic files, pre-process the dataset lastly.
        """
        self.logger.debug(f'Loading {self.__class__} from scratch')

        self._get_preset()
        self._get_field_from_config()
        self._load_data(self.dataset_name, self.dataset_path)

        self._data_processing()
        #print("after!!!!:", self.inter_feat)
        #self._get_popularity()

        self._change_feat_format()
        self.join(self.inter_feat)
        #print("after join:", self.inter_feat)
        if self.config['split_mode'] == 4:
            self._reclass_genre()

    def _get_preset(self):
        """Initialization useful inside attributes.
        """
        self.dataset_path = self.config['data_path']

        self.field2type = {}
        self.field2source = {}
        self.field2id_token = {}
        self.field2token_id = {}
        self.field2seqlen = self.config['seq_len'] or {}
        self._preloaded_weight = {}
        self.benchmark_filename_list = self.config['benchmark_filename']

    def _get_field_from_config(self):
        """Initialization common field names.
        """
        self.uid_field = self.config['USER_ID_FIELD'] #session_id
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.label_field = self.config['LABEL_FIELD']
        self.time_field = self.config['TIME_FIELD']

        if (self.uid_field is None) ^ (self.iid_field is None):
            raise ValueError(
                'USER_ID_FIELD and ITEM_ID_FIELD need to be set at the same time or not set at the same time.'
            )

        self.logger.debug(f'uid_field: {self.uid_field}')
        self.logger.debug(f'iid_field: {self.iid_field}')

    def _data_processing(self):
        """Data preprocessing, including:

        - Data filtering
        - Remap ID
        - Missing value imputation
        - Normalization
        - Preloading weights initialization
        """
        self.feat_name_list = self._build_feat_name_list()
        #print("self.feat_name_list:", self.feat_name_list)

        if self.benchmark_filename_list is None:
            self._data_filtering()
            #print("data_filtering:", self.item_feat)

        self._remap_ID_all()
        #print("after remap~~~~:", self.item_feat)
        self._user_item_feat_preparation()
        #print("2:", self.inter_feat)
        self._fill_nan()
        #print("3:", self.inter_feat)
        self._set_label_by_threshold()
        #print("4:", self.inter_feat)
        self._normalize()
        #print("5:", self.inter_feat)
        self._preload_weight_matrix()
        #print("6:", self.inter_feat)
        ############################
        self._get_popularity()

    def _get_popularity(self):
        item_inter_num = Counter(self.inter_feat[self.iid_field].values)
        self.item_inter_num = item_inter_num

    def _reclass_genre(self):
        g_class_num = 8
        #g_class = {1: [1, 2, 15], 2: [3], 3: [4, 8, 14, 18], 4:[5, 12], 5:[6], 6:[7, 17],
        #           7: [9, 10, 11, 16], 8:[13]}
        # for c, gc in g_class.items():
        #    print(c)
        #    for id in gc:
        #        print(self.gener_mp[id - 1])
        g_map =[0, 1, 1, 2, 3, 4, 5, 6, 3, 7, 7, 7, 4, 8, 3, 1, 7, 6, 3]

        origin_genre_list = self.inter_feat['genre']
        new_genre_list = torch.zeros_like(origin_genre_list)
        for gid, origin_genre in enumerate(origin_genre_list):
            #print("origin:::",origin_genre)
            new_genre = []
            for origin in origin_genre:
                if origin == 0:
                    break
                new_genre.append(g_map[origin])
            new_genre = list(set(new_genre))

            while (len(new_genre) < len(origin_genre)):
                new_genre.append(0)
            #print("new_genre:", new_genre)
            new_genre_list[gid] = torch.tensor(new_genre)
        print("new_genre_list::::",new_genre_list)

        new_list = Interaction({'new_genre': new_genre_list})

        self.inter_feat.update(new_list)
        print(self.inter_feat['new_genre'])

    def _data_filtering(self):
        """Data filtering

        - Filter missing user_id or item_id
        - Remove duplicated user-item interaction
        - Value-based data filtering
        - Remove interaction by user or item
        - K-core data filtering

        Note:
            After filtering, feats(``DataFrame``) has non-continuous index,
            thus :meth:`~recbole.data.dataset.dataset.Dataset._reset_index` will reset the index of feats.
        """
        self._filter_nan_user_or_item()
        #print("1:", self.inter_feat)
        self._remove_duplication()
        #print("2:", self.inter_feat)
        self._filter_by_field_value()
        #print("3:", self.inter_feat)
        self._filter_inter_by_user_or_item()
        #print("4:", self.inter_feat)
        self._filter_by_inter_num()
        #print("5:", self.inter_feat)
        self._reset_index()

    def _build_feat_name_list(self):
        """Feat list building.

        Any feat loaded by Dataset can be found in ``feat_name_list``

        Returns:
            built feature name list.

        Note:
            Subclasses can inherit this method to add new feat.
        """
        feat_name_list = [
            feat_name for feat_name in ['inter_feat', 'user_feat', 'item_feat']
            if getattr(self, feat_name, None) is not None
        ]
        if self.config['additional_feat_suffix'] is not None:
            for suf in self.config['additional_feat_suffix']:
                if getattr(self, f'{suf}_feat', None) is not None:
                    feat_name_list.append(f'{suf}_feat')
        return feat_name_list

    def _restore_saved_dataset(self, saved_dataset):
        """Restore saved dataset from ``saved_dataset``.

        Args:
            saved_dataset (str): path for the saved dataset.
        """
        self.logger.debug(f'Restoring dataset from [{saved_dataset}].')

        if (saved_dataset is None) or (not os.path.isdir(saved_dataset)):
            raise ValueError(f'Filepath [{saved_dataset}] need to be a dir.')

        with open(os.path.join(saved_dataset, 'basic-info.json')) as file:
            basic_info = json.load(file)

        for k in basic_info:
            setattr(self, k, basic_info[k])

        feats = ['inter', 'user', 'item']
        for name in feats:
            cur_file_name = os.path.join(saved_dataset, f'{name}.csv')
            if os.path.isfile(cur_file_name):
                df = pd.read_csv(cur_file_name)
                setattr(self, f'{name}_feat', df)
            else:
                setattr(self, f'{name}_feat', None)

        self._get_field_from_config()

    def _load_data(self, token, dataset_path):
        """Load features.

        Firstly load interaction features, then user/item features optionally,
        finally load additional features if ``config['additional_feat_suffix']`` is set.

        Args:
            token (str): dataset name.
            dataset_path (str): path of dataset dir.
        """
        self._load_inter_feat(token, dataset_path)
        self.user_feat = self._load_user_or_item_feat(token, dataset_path, FeatureSource.USER, 'uid_field')
        self.item_feat = self._load_user_or_item_feat(token, dataset_path, FeatureSource.ITEM, 'iid_field')
        #print("load_item!!!!!!!!", self.item_feat)
        self._load_additional_feat(token, dataset_path)

    def _load_inter_feat(self, token, dataset_path):
        """Load interaction features.

        If ``config['benchmark_filename']`` is not set, load interaction features from ``.inter``.

        Otherwise, load interaction features from a file list, named ``dataset_name.xxx.inter``,
        where ``xxx`` if from ``config['benchmark_filename']``.
        After loading, ``self.file_size_list`` stores the length of each interaction file.

        Args:
            token (str): dataset name.
            dataset_path (str): path of dataset dir.
        """
        if self.benchmark_filename_list is None:
            inter_feat_path = os.path.join(dataset_path, f'{token}.inter') #dataset/yoochoose/yoochoose.inter
            if not os.path.isfile(inter_feat_path):
                raise ValueError(f'File {inter_feat_path} not exist.')
            inter_feat = self._load_feat(inter_feat_path, FeatureSource.INTERACTION)
            self.logger.debug(f'Interaction feature loaded successfully from [{inter_feat_path}].')
            self.inter_feat = inter_feat
            #print("self.inter_feat:",self.inter_feat)
        else:
            sub_inter_lens = []
            sub_inter_feats = []
            for filename in self.benchmark_filename_list:
                file_path = os.path.join(dataset_path, f'{token}.{filename}.inter')
                if os.path.isfile(file_path):
                    temp = self._load_feat(file_path, FeatureSource.INTERACTION)
                    sub_inter_feats.append(temp)
                    sub_inter_lens.append(len(temp))
                else:
                    raise ValueError(f'File {file_path} not exist.')
            inter_feat = pd.concat(sub_inter_feats)
            self.inter_feat, self.file_size_list = inter_feat, sub_inter_lens

    def _load_user_or_item_feat(self, token, dataset_path, source, field_name):
        """Load user/item features.

        Args:
            token (str): dataset name.
            dataset_path (str): path of dataset dir.
            source (FeatureSource): source of user/item feature.
            field_name (str): ``uid_field`` or ``iid_field``

        Returns:
            pandas.DataFrame: Loaded feature

        Note:
            ``user_id`` and ``item_id`` has source :obj:`~recbole.utils.enum_type.FeatureSource.USER_ID` and
            :obj:`~recbole.utils.enum_type.FeatureSource.ITEM_ID`
        """
        feat_path = os.path.join(dataset_path, f'{token}.{source.value}')
        if os.path.isfile(feat_path):
            feat = self._load_feat(feat_path, source)
            self.logger.debug(f'[{source.value}] feature loaded successfully from [{feat_path}].')
        else:
            feat = None
            self.logger.debug(f'[{feat_path}] not found, [{source.value}] features are not loaded.')

        field = getattr(self, field_name, None)
        if feat is not None and field is None:
            raise ValueError(f'{field_name} must be exist if {source.value}_feat exist.')
        if feat is not None and field not in feat:
            raise ValueError(f'{field_name} must be loaded if {source.value}_feat is loaded.')
        if field in self.field2source:
            self.field2source[field] = FeatureSource(source.value + '_id')
        #print("after#####################################", self.field2source)
        return feat

    def _load_additional_feat(self, token, dataset_path):
        """Load additional features.

        For those additional features, e.g. pretrained entity embedding, user can set them
        as ``config['additional_feat_suffix']``, then they will be loaded and stored in
        :attr:`feat_name_list`. See :doc:`../user_guide/data/data_args` for details.

        Args:
            token (str): dataset name.
            dataset_path (str): path of dataset dir.
        """
        if self.config['additional_feat_suffix'] is None:
            return
        for suf in self.config['additional_feat_suffix']:
            if hasattr(self, f'{suf}_feat'):
                raise ValueError(f'{suf}_feat already exist.')
            feat_path = os.path.join(dataset_path, f'{token}.{suf}')
            if os.path.isfile(feat_path):
                feat = self._load_feat(feat_path, suf)
            else:
                raise ValueError(f'Additional feature file [{feat_path}] not found.')
            setattr(self, f'{suf}_feat', feat)

    def _get_load_and_unload_col(self, source):
        """Parsing ``config['load_col']`` and ``config['unload_col']`` according to source.
        See :doc:`../user_guide/data/data_args` for detail arg setting.

        Args:
            source (FeatureSource): source of input file.

        Returns:
            tuple: tuple of parsed ``load_col`` and ``unload_col``, see :doc:`../user_guide/data/data_args` for details.
        """
        if isinstance(source, FeatureSource):
            source = source.value #inter
        if self.config['load_col'] is None:
            load_col = None
        elif source not in self.config['load_col']:
            load_col = set()
        elif self.config['load_col'][source] == '*':
            load_col = None
        else:
            load_col = set(self.config['load_col'][source])

        if self.config['unload_col'] is not None and source in self.config['unload_col']:
            unload_col = set(self.config['unload_col'][source])
        else:
            unload_col = None

        if load_col and unload_col:
            raise ValueError(f'load_col [{load_col}] and unload_col [{unload_col}] can not be set the same time.')

        self.logger.debug(f'[{source}]: ')
        self.logger.debug(f'\t load_col: [{load_col}]')
        self.logger.debug(f'\t unload_col: [{unload_col}]')
        return load_col, unload_col

    def _load_feat(self, filepath, source):
        """Load features according to source into :class:`pandas.DataFrame`.

        Set features' properties, e.g. type, source and length.

        Args:
            filepath (str): path of input file.
            source (FeatureSource or str): source of input file.

        Returns:
            pandas.DataFrame: Loaded feature

        Note:
            For sequence features, ``seqlen`` will be loaded, but data in DataFrame will not be cut off.
            Their length is limited only after calling :meth:`~_dict_to_interaction` or
            :meth:`~_dataframe_to_interaction`
        """
        self.logger.debug(f'Loading feature from [{filepath}] (source: [{source}]).')

        load_col, unload_col = self._get_load_and_unload_col(source)
        if load_col == set():
            return None

        field_separator = self.config['field_separator']
        columns = []
        usecols = []
        dtype = {}
        with open(filepath, 'r') as f:
            head = f.readline()[:-1]
        for field_type in head.split(field_separator):
            field, ftype = field_type.split(':')
            try:
                ftype = FeatureType(ftype)
            except ValueError:
                raise ValueError(f'Type {ftype} from field {field} is not supported.')
            if load_col is not None and field not in load_col:
                continue
            if unload_col is not None and field in unload_col:
                continue
            if isinstance(source, FeatureSource) or source != 'link':
                self.field2source[field] = source
                self.field2type[field] = ftype
                if not ftype.value.endswith('seq'):
                    self.field2seqlen[field] = 1
            columns.append(field)
            usecols.append(field_type)
            dtype[field_type] = np.float64 if ftype == FeatureType.FLOAT else str

        if len(columns) == 0:
            self.logger.warning(f'No columns has been loaded from [{source}]')
            return None
        #print("##################################################",usecols)
        df = pd.read_csv(filepath, delimiter=self.config['field_separator'], usecols=usecols, dtype=dtype)
        df.columns = columns

        seq_separator = self.config['seq_separator']
        for field in columns:
            ftype = self.field2type[field]
            if not ftype.value.endswith('seq'):
                continue
            df[field].fillna(value='', inplace=True)
            if ftype == FeatureType.TOKEN_SEQ:
                df[field] = [list(filter(None, _.split(seq_separator))) for _ in df[field].values]
            elif ftype == FeatureType.FLOAT_SEQ:
                df[field] = [list(map(float, filter(None, _.split(seq_separator)))) for _ in df[field].values]
            self.field2seqlen[field] = max(map(len, df[field].values))
        return df

    def _user_item_feat_preparation(self):
        """Sort :attr:`user_feat` and :attr:`item_feat` by ``user_id`` or ``item_id``.
        Missing values will be filled later.
        """
        if self.user_feat is not None:
            new_user_df = pd.DataFrame({self.uid_field: np.arange(self.user_num)})
            self.user_feat = pd.merge(new_user_df, self.user_feat, on=self.uid_field, how='left')
            self.logger.debug('ordering user features by user id.')
        if self.item_feat is not None:
            new_item_df = pd.DataFrame({self.iid_field: np.arange(self.item_num)})
            self.item_feat = pd.merge(new_item_df, self.item_feat, on=self.iid_field, how='left')
            self.logger.debug('ordering item features by user id.')

    def _preload_weight_matrix(self):
        """Transfer preload weight features into :class:`numpy.ndarray` with shape ``[id_token_length]``
        or ``[id_token_length, seqlen]``. See :doc:`../user_guide/data/data_args` for detail arg setting.
        """
        preload_fields = self.config['preload_weight']
        if preload_fields is None:
            return

        self.logger.debug(f'Preload weight matrix for {preload_fields}.')

        for preload_id_field in preload_fields:
            preload_value_field = preload_fields[preload_id_field]
            if preload_id_field not in self.field2source:
                raise ValueError(f'Preload id field [{preload_id_field}] not exist.')
            if preload_value_field not in self.field2source:
                raise ValueError(f'Preload value field [{preload_value_field}] not exist.')
            pid_source = self.field2source[preload_id_field]
            pv_source = self.field2source[preload_value_field]
            if pid_source != pv_source:
                raise ValueError(
                    f'Preload id field [{preload_id_field}] is from source [{pid_source}],'
                    f'while preload value field [{preload_value_field}] is from source [{pv_source}], '
                    f'which should be the same.'
                )
            for feat_name in self.feat_name_list:
                feat = getattr(self, feat_name)
                if preload_id_field in feat:
                    id_ftype = self.field2type[preload_id_field]
                    if id_ftype != FeatureType.TOKEN:
                        raise ValueError(
                            f'Preload id field [{preload_id_field}] should be type token, but is [{id_ftype}].'
                        )
                    value_ftype = self.field2type[preload_value_field]
                    token_num = self.num(preload_id_field)
                    if value_ftype == FeatureType.FLOAT:
                        matrix = np.zeros(token_num)
                        preload_ids = feat[preload_id_field].values
                        preload_values = feat[preload_value_field].values
                        for pid, pv in zip(preload_ids, preload_values):
                            matrix[pid] = pv
                    elif value_ftype == FeatureType.FLOAT_SEQ:
                        max_len = self.field2seqlen[preload_value_field]
                        matrix = np.zeros((token_num, max_len))
                        preload_ids = feat[preload_id_field].values
                        preload_values = feat[preload_value_field].to_list()
                        for pid, prow in zip(preload_ids, preload_values):
                            length = len(prow)
                            if length <= max_len:
                                matrix[pid, :length] = prow
                            else:
                                matrix[pid] = prow[:max_len]
                    else:
                        self.logger.warning(
                            f'Field [{preload_value_field}] with type [{value_ftype}] is not `float` or `float_seq`, '
                            f'which will not be handled by preload matrix.'
                        )
                        continue
                    self._preloaded_weight[preload_id_field] = matrix

    def _fill_nan(self):
        """Missing value imputation.

        For fields with type :obj:`~recbole.utils.enum_type.FeatureType.TOKEN`, missing value will be filled by
        ``[PAD]``, which indexed as 0.

        For fields with type :obj:`~recbole.utils.enum_type.FeatureType.FLOAT`, missing value will be filled by
        the average of original data.
        """
        self.logger.debug('Filling nan')

        for feat_name in self.feat_name_list:
            feat = getattr(self, feat_name)
            for field in feat:
                ftype = self.field2type[field]
                if ftype == FeatureType.TOKEN:
                    feat[field].fillna(value=0, inplace=True)
                elif ftype == FeatureType.FLOAT:
                    feat[field].fillna(value=feat[field].mean(), inplace=True)
                else:
                    feat[field] = feat[field].apply(lambda x: [] if isinstance(x, float) else x)

    def _normalize(self):
        """Normalization if ``config['normalize_field']`` or ``config['normalize_all']`` is set.
        See :doc:`../user_guide/data/data_args` for detail arg setting.

        .. math::
            x' = \frac{x - x_{min}}{x_{max} - x_{min}}

        Note:
            Only float-like fields can be normalized.
        """
        if self.config['normalize_field'] is not None and self.config['normalize_all'] is True:
            raise ValueError('Normalize_field and normalize_all can\'t be set at the same time.')

        if self.config['normalize_field']:
            fields = self.config['normalize_field']
            for field in fields:
                ftype = self.field2type[field]
                if field not in self.field2type:
                    raise ValueError(f'Field [{field}] does not exist.')
                elif ftype != FeatureType.FLOAT and ftype != FeatureType.FLOAT_SEQ:
                    self.logger.warning(f'{field} is not a FLOAT/FLOAT_SEQ feat, which will not be normalized.')
        elif self.config['normalize_all']:
            fields = self.float_like_fields
        else:
            return

        self.logger.debug(f'Normalized fields: {fields}')

        for feat_name in self.feat_name_list:
            feat = getattr(self, feat_name)
            for field in feat:
                if field not in fields:
                    continue
                ftype = self.field2type[field]
                if ftype == FeatureType.FLOAT:
                    lst = feat[field].values
                    mx, mn = max(lst), min(lst)
                    if mx == mn:
                        self.logger.warning(f'All the same value in [{field}] from [{feat}_feat].')
                        feat[field] = 1.0
                    else:
                        feat[field] = (lst - mn) / (mx - mn)
                elif ftype == FeatureType.FLOAT_SEQ:
                    split_point = np.cumsum(feat[field].agg(len))[:-1]
                    lst = feat[field].agg(np.concatenate)
                    mx, mn = max(lst), min(lst)
                    if mx == mn:
                        self.logger.warning(f'All the same value in [{field}] from [{feat}_feat].')
                        lst = 1.0
                    else:
                        lst = (lst - mn) / (mx - mn)
                    lst = np.split(lst, split_point)
                    feat[field] = lst

    def _filter_nan_user_or_item(self):
        """Filter NaN user_id and item_id
        """
        for field, name in zip([self.uid_field, self.iid_field], ['user', 'item']):
            feat = getattr(self, name + '_feat')
            if feat is not None:
                dropped_feat = feat.index[feat[field].isnull()]
                if len(dropped_feat):
                    self.logger.warning(
                        f'In {name}_feat, line {list(dropped_feat + 2)}, {field} do not exist, so they will be removed.'
                    )
                    feat.drop(feat.index[dropped_feat], inplace=True)
            if field is not None:
                dropped_inter = self.inter_feat.index[self.inter_feat[field].isnull()]
                if len(dropped_inter):
                    self.logger.warning(
                        f'In inter_feat, line {list(dropped_inter + 2)}, {field} do not exist, so they will be removed.'
                    )
                    self.inter_feat.drop(self.inter_feat.index[dropped_inter], inplace=True)

    def _remove_duplication(self):
        """Remove duplications in inter_feat.

        If :attr:`self.config['rm_dup_inter']` is not ``None``, it will remove duplicated user-item interactions.

        Note:
            Before removing duplicated user-item interactions, if :attr:`time_field` existed, :attr:`inter_feat`
            will be sorted by :attr:`time_field` in ascending order.
        """
        keep = self.config['rm_dup_inter']
        if keep is None:
            return
        self._check_field('uid_field', 'iid_field')

        if self.time_field in self.inter_feat:
            self.inter_feat.sort_values(by=[self.time_field], ascending=True, inplace=True)
            self.logger.info(
                f'Records in original dataset have been sorted by value of [{self.time_field}] in ascending order.'
            )
        else:
            self.logger.warning(
                f'Timestamp field has not been loaded or specified, '
                f'thus strategy [{keep}] of duplication removal may be meaningless.'
            )
        self.inter_feat.drop_duplicates(subset=[self.uid_field, self.iid_field], keep=keep, inplace=True)

    def _filter_by_inter_num(self):
        """Filter by number of interaction.

        Upper/Lower bounds can be set, only users/items between upper/lower bounds can be remained.
        See :doc:`../user_guide/data/data_args` for detail arg setting.

        Note:
            Lower bound is also called k-core filtering, which means this method will filter loops
            until all the users and items has at least k interactions.
        """
        if self.uid_field is None or self.iid_field is None:
            return

        max_user_inter_num = self.config['max_user_inter_num']
        min_user_inter_num = self.config['min_user_inter_num']
        max_item_inter_num = self.config['max_item_inter_num']
        min_item_inter_num = self.config['min_item_inter_num']

        if max_user_inter_num is None and min_user_inter_num is None:
            user_inter_num = Counter()
        else:
            user_inter_num = Counter(self.inter_feat[self.uid_field].values)

        if max_item_inter_num is None and min_item_inter_num is None:
            item_inter_num = Counter()
        else:
            item_inter_num = Counter(self.inter_feat[self.iid_field].values)

        while True:
            ban_users = self._get_illegal_ids_by_inter_num(
                field=self.uid_field,
                feat=self.user_feat,
                inter_num=user_inter_num,
                max_num=max_user_inter_num,
                min_num=min_user_inter_num
            )
            ban_items = self._get_illegal_ids_by_inter_num(
                field=self.iid_field,
                feat=self.item_feat,
                inter_num=item_inter_num,
                max_num=max_item_inter_num,
                min_num=min_item_inter_num
            )

            if len(ban_users) == 0 and len(ban_items) == 0:
                break

            if self.user_feat is not None:
                dropped_user = self.user_feat[self.uid_field].isin(ban_users)
                self.user_feat.drop(self.user_feat.index[dropped_user], inplace=True)

            if self.item_feat is not None:
                dropped_item = self.item_feat[self.iid_field].isin(ban_items)
                self.item_feat.drop(self.item_feat.index[dropped_item], inplace=True)

            dropped_inter = pd.Series(False, index=self.inter_feat.index)
            user_inter = self.inter_feat[self.uid_field]
            item_inter = self.inter_feat[self.iid_field]
            dropped_inter |= user_inter.isin(ban_users)
            dropped_inter |= item_inter.isin(ban_items)

            user_inter_num -= Counter(user_inter[dropped_inter].values)
            item_inter_num -= Counter(item_inter[dropped_inter].values)

            dropped_index = self.inter_feat.index[dropped_inter]
            self.logger.debug(f'[{len(dropped_index)}] dropped interactions.')
            self.inter_feat.drop(dropped_index, inplace=True)

    def _get_illegal_ids_by_inter_num(self, field, feat, inter_num, max_num=None, min_num=None):
        """Given inter feat, return illegal ids, whose inter num out of [min_num, max_num]

        Args:
            field (str): field name of user_id or item_id.
            feat (pandas.DataFrame): interaction feature.
            inter_num (Counter): interaction number counter.
            max_num (int, optional): max number of interaction. Defaults to ``None``.
            min_num (int, optional): min number of interaction. Defaults to ``None``.

        Returns:
            set: illegal ids, whose inter num out of [min_num, max_num]
        """
        self.logger.debug(f'get_illegal_ids_by_inter_num: field=[{field}], max_num=[{max_num}], min_num=[{min_num}]')

        max_num = max_num or np.inf
        min_num = min_num or -1

        ids = {id_ for id_ in inter_num if inter_num[id_] < min_num or inter_num[id_] > max_num}

        if feat is not None:
            for id_ in feat[field].values:
                if inter_num[id_] < min_num:
                    ids.add(id_)
        self.logger.debug(f'[{len(ids)}] illegal_ids_by_inter_num, field=[{field}]')
        return ids

    def _filter_by_field_value(self):
        """Filter features according to its values.
        """
        filter_field = []
        filter_field += self._drop_by_value(self.config['lowest_val'], lambda x, y: x < y)
        filter_field += self._drop_by_value(self.config['highest_val'], lambda x, y: x > y)
        filter_field += self._drop_by_value(self.config['equal_val'], lambda x, y: x != y)
        filter_field += self._drop_by_value(self.config['not_equal_val'], lambda x, y: x == y)

    def _reset_index(self):
        """Reset index for all feats in :attr:`feat_name_list`.
        """
        for feat_name in self.feat_name_list:
            feat = getattr(self, feat_name)
            if feat.empty:
                raise ValueError('Some feat is empty, please check the filtering settings.')
            feat.reset_index(drop=True, inplace=True)

    def _drop_by_value(self, val, cmp):
        """Drop illegal rows by value.

        Args:
            val (dict): value that compared to.
            cmp (Callable): return False if a row need to be dropped

        Returns:
            field names that used to compare with val.
        """
        if val is None:
            return []

        self.logger.debug(f'drop_by_value: val={val}')
        filter_field = []
        for field in val:
            if field not in self.field2type:
                raise ValueError(f'Field [{field}] not defined in dataset.')
            if self.field2type[field] not in {FeatureType.FLOAT, FeatureType.FLOAT_SEQ}:
                raise ValueError(f'Field [{field}] is not float-like field in dataset, which can\'t be filter.')
            for feat_name in self.feat_name_list:
                feat = getattr(self, feat_name)
                if field in feat:
                    feat.drop(feat.index[cmp(feat[field].values, val[field])], inplace=True)
            filter_field.append(field)
        return filter_field

    def _del_col(self, feat, field):
        """Delete columns

        Args:
            feat (pandas.DataFrame or Interaction): the feat contains field.
            field (str): field name to be dropped.
        """
        self.logger.debug(f'Delete column [{field}].')
        if isinstance(feat, Interaction):
            feat.drop(column=field)
        else:
            feat.drop(columns=field, inplace=True)
        for dct in [self.field2id_token, self.field2token_id, self.field2seqlen, self.field2source, self.field2type]:
            if field in dct:
                del dct[field]

    def _filter_inter_by_user_or_item(self):
        """Remove interaction in inter_feat which user or item is not in user_feat or item_feat.
        """
        if self.config['filter_inter_by_user_or_item'] is not True:
            return

        remained_inter = pd.Series(True, index=self.inter_feat.index)

        if self.user_feat is not None:
            remained_uids = self.user_feat[self.uid_field].values
            remained_inter &= self.inter_feat[self.uid_field].isin(remained_uids)

        if self.item_feat is not None:
            remained_iids = self.item_feat[self.iid_field].values
            remained_inter &= self.inter_feat[self.iid_field].isin(remained_iids)

        self.inter_feat.drop(self.inter_feat.index[~remained_inter], inplace=True)

    def _set_label_by_threshold(self):
        """Generate 0/1 labels according to value of features.

        According to ``config['threshold']``, those rows with value lower than threshold will
        be given negative label, while the other will be given positive label.
        See :doc:`../user_guide/data/data_args` for detail arg setting.

        Note:
            Key of ``config['threshold']`` if a field name.
            This field will be dropped after label generation.
        """
        threshold = self.config['threshold']
        if threshold is None:
            return

        self.logger.debug(f'Set label by {threshold}.')

        if len(threshold) != 1:
            raise ValueError('Threshold length should be 1.')

        self.set_field_property(self.label_field, FeatureType.FLOAT, FeatureSource.INTERACTION, 1)
        for field, value in threshold.items():
            if field in self.inter_feat:
                self.inter_feat[self.label_field] = (self.inter_feat[field] >= value).astype(int)
            else:
                raise ValueError(f'Field [{field}] not in inter_feat.')
            self._del_col(self.inter_feat, field)

    def _get_fields_in_same_space(self):
        """Parsing ``config['fields_in_same_space']``. See :doc:`../user_guide/data/data_args` for detail arg setting.

        Note:
            - Each field can only exist ONCE in ``config['fields_in_same_space']``.
            - user_id and item_id can not exist in ``config['fields_in_same_space']``.
            - only token-like fields can exist in ``config['fields_in_same_space']``.
        """
        fields_in_same_space = self.config['fields_in_same_space'] or []
        fields_in_same_space = [set(_) for _ in fields_in_same_space]
        additional = []
        token_like_fields = self.token_like_fields
        for field in token_like_fields:
            count = 0
            for field_set in fields_in_same_space:
                if field in field_set:
                    count += 1
            if count == 0:
                additional.append({field})
            elif count == 1:
                continue
            else:
                raise ValueError(f'Field [{field}] occurred in `fields_in_same_space` more than one time.')

        for field_set in fields_in_same_space:
            if self.uid_field in field_set and self.iid_field in field_set:
                raise ValueError('uid_field and iid_field can\'t in the same ID space')
            for field in field_set:
                if field not in token_like_fields:
                    raise ValueError(f'Field [{field}] is not a token-like field.')

        fields_in_same_space.extend(additional)
        return fields_in_same_space

    def _get_remap_list(self, field_set):
        """Transfer set of fields in the same remapping space into remap list.

        If ``uid_field`` or ``iid_field`` in ``field_set``,
        field in :attr:`inter_feat` will be remapped firstly,
        then field in :attr:`user_feat` or :attr:`item_feat` will be remapped next, finally others.

        Args:
            field_set (set): Set of fields in the same remapping space

        Returns:
            list:
            - feat (pandas.DataFrame)
            - field (str)
            - ftype (FeatureType)

            They will be concatenated in order, and remapped together.
        """
        remap_list = []
        for field, feat in zip([self.uid_field, self.iid_field], [self.user_feat, self.item_feat]):
            if field in field_set:
                field_set.remove(field)
                remap_list.append((self.inter_feat, field, FeatureType.TOKEN))
                if feat is not None:
                    remap_list.append((feat, field, FeatureType.TOKEN))
        for field in field_set:
            source = self.field2source[field]
            if isinstance(source, FeatureSource):
                source = source.value
            feat = getattr(self, f'{source}_feat')
            ftype = self.field2type[field]
            remap_list.append((feat, field, ftype))
        return remap_list

    def _remap_ID_all(self):
        """Get ``config['fields_in_same_space']`` firstly, and remap each.
        """
        fields_in_same_space = self._get_fields_in_same_space()
        #print("&&&&&&&&&", fields_in_same_space)
        self.logger.debug(f'fields_in_same_space: {fields_in_same_space}')
        for field_set in fields_in_same_space:
            remap_list = self._get_remap_list(field_set)
            #print("remap_list!!!!!!:", remap_list)
            self._remap(remap_list)

    def _concat_remaped_tokens(self, remap_list):
        """Given ``remap_list``, concatenate values in order.

        Args:
            remap_list (list): See :meth:`_get_remap_list` for detail.

        Returns:
            tuple: tuple of:
            - tokens after concatenation.
            - split points that can be used to restore the concatenated tokens.
        """
        tokens = []
        for feat, field, ftype in remap_list:
            if ftype == FeatureType.TOKEN:
                tokens.append(feat[field].values)
            elif ftype == FeatureType.TOKEN_SEQ:
                tokens.append(feat[field].agg(np.concatenate))
        split_point = np.cumsum(list(map(len, tokens)))[:-1]
        tokens = np.concatenate(tokens)
        return tokens, split_point

    def _remap(self, remap_list):
        """Remap tokens using :meth:`pandas.factorize`.

        Args:
            remap_list (list): See :meth:`_get_remap_list` for detail.
        """
        tokens, split_point = self._concat_remaped_tokens(remap_list)
        #print("token:", tokens)
        #print("split_point:", split_point)
        new_ids_list, mp = pd.factorize(tokens)
        for feat, field, ftype in remap_list:
            if field == 'user_id':
                self.user_mp = mp
            elif field == 'item_id':
                self.item_mp = mp
            else:
                self.gener_mp = mp
        print("genre:::", mp)

        new_ids_list = np.split(new_ids_list + 1, split_point)
        #print("new_ids_list:", new_ids_list)
        mp = np.array(['[PAD]'] + list(mp))
        token_id = {t: i for i, t in enumerate(mp)}

        for (feat, field, ftype), new_ids in zip(remap_list, new_ids_list):
            if field not in self.field2id_token:
                self.field2id_token[field] = mp
                self.field2token_id[field] = token_id
            if ftype == FeatureType.TOKEN:
                feat[field] = new_ids
            elif ftype == FeatureType.TOKEN_SEQ:
                split_point = np.cumsum(feat[field].agg(len))[:-1]
                feat[field] = np.split(new_ids, split_point)

    def _change_feat_format(self):
        """Change feat format from :class:`pandas.DataFrame` to :class:`Interaction`.
        """
        for feat_name in self.feat_name_list:
            #print("feat_name:", feat_name)
            feat = getattr(self, feat_name)
            #print("feat:", feat)
            setattr(self, feat_name, self._dataframe_to_interaction(feat))
            #print("after!!!!!")

    @dlapi.set()
    def num(self, field):
        """Given ``field``, for token-like fields, return the number of different tokens after remapping,
        for float-like fields, return ``1``.

        Args:
            field (str): field name to get token number.

        Returns:
            int: The number of different tokens (``1`` if ``field`` is a float-like field).
        """
        if field not in self.field2type:
            raise ValueError(f'Field [{field}] not defined in dataset.')
        if self.field2type[field] not in {FeatureType.TOKEN, FeatureType.TOKEN_SEQ}:
            return self.field2seqlen[field]
        else:
            return len(self.field2id_token[field])

    @dlapi.set()
    def fields(self, ftype=None):
        """Given type of features, return all the field name of this type.
        if ``ftype = None``, return all the fields.

        Args:
            ftype (FeatureType, optional): Type of features.

        Returns:
            list: List of field names.
        """
        ftype = set(ftype) if ftype is not None else set(FeatureType)
        ret = []
        for field in self.field2type:
            tp = self.field2type[field]
            if tp in ftype:
                ret.append(field)
        return ret

    @property
    def float_like_fields(self):
        """Get fields of type :obj:`~recbole.utils.enum_type.FeatureType.FLOAT` and
        :obj:`~recbole.utils.enum_type.FeatureType.FLOAT_SEQ`.

        Returns:
            list: List of field names.
        """
        return self.fields([FeatureType.FLOAT, FeatureType.FLOAT_SEQ])

    @property
    def token_like_fields(self):
        """Get fields of type :obj:`~recbole.utils.enum_type.FeatureType.TOKEN` and
        :obj:`~recbole.utils.enum_type.FeatureType.TOKEN_SEQ`.

        Returns:
            list: List of field names.
        """
        return self.fields([FeatureType.TOKEN, FeatureType.TOKEN_SEQ])

    @property
    def seq_fields(self):
        """Get fields of type :obj:`~recbole.utils.enum_type.FeatureType.TOKEN_SEQ` and
        :obj:`~recbole.utils.enum_type.FeatureType.FLOAT_SEQ`.

        Returns:
            list: List of field names.
        """
        return self.fields([FeatureType.FLOAT_SEQ, FeatureType.TOKEN_SEQ])

    @property
    def non_seq_fields(self):
        """Get fields of type :obj:`~recbole.utils.enum_type.FeatureType.TOKEN` and
        :obj:`~recbole.utils.enum_type.FeatureType.FLOAT`.

        Returns:
            list: List of field names.
        """
        return self.fields([FeatureType.FLOAT, FeatureType.TOKEN])

    def set_field_property(self, field, field_type, field_source, field_seqlen):
        """Set a new field's properties.

        Args:
            field (str): Name of the new field.
            field_type (FeatureType): Type of the new field.
            field_source (FeatureSource): Source of the new field.
            field_seqlen (int): max length of the sequence in ``field``.
                ``1`` if ``field``'s type is not sequence-like.
        """
        self.field2type[field] = field_type
        self.field2source[field] = field_source
        self.field2seqlen[field] = field_seqlen

    def copy_field_property(self, dest_field, source_field):
        """Copy properties from ``dest_field`` towards ``source_field``.

        Args:
            dest_field (str): Destination field.
            source_field (str): Source field.
        """
        self.field2type[dest_field] = self.field2type[source_field]
        self.field2source[dest_field] = self.field2source[source_field]
        self.field2seqlen[dest_field] = self.field2seqlen[source_field]

    @dlapi.set()
    def token2id(self, field, tokens):
        """Map external tokens to internal ids.

        Args:
            field (str): Field of external tokens.
            tokens (str, list or numpy.ndarray): External tokens.

        Returns:
            int or numpy.ndarray: The internal ids of external tokens.
        """
        if isinstance(tokens, str):
            if tokens in self.field2token_id[field]:
                return self.field2token_id[field][tokens]
            else:
                raise ValueError('token [{}] is not existed')
        elif isinstance(tokens, (list, np.ndarray)):
            return np.array([self.token2id(field, token) for token in tokens])
        else:
            raise TypeError('The type of tokens [{}] is not supported')

    @dlapi.set()
    def id2token(self, field, ids):
        """Map internal ids to external tokens.

        Args:
            field (str): Field of internal ids.
            ids (int, list, numpy.ndarray or torch.Tensor): Internal ids.

        Returns:
            str or numpy.ndarray: The external tokens of internal ids.
        """
        try:
            return self.field2id_token[field][ids]
        except IndexError:
            if isinstance(ids, list):
                raise ValueError(f'[{ids}] is not a one-dimensional list.')
            else:
                raise ValueError(f'[{ids}] is not a valid ids.')

    @property
    @dlapi.set()
    def user_num(self):
        """Get the number of different tokens of ``self.uid_field``.

        Returns:
            int: Number of different tokens of ``self.uid_field``.
        """
        self._check_field('uid_field')
        return self.num(self.uid_field)

    @property
    @dlapi.set()
    def item_num(self):
        """Get the number of different tokens of ``self.iid_field``.

        Returns:
            int: Number of different tokens of ``self.iid_field``.
        """
        self._check_field('iid_field')
        return self.num(self.iid_field)

    @property
    def inter_num(self):
        """Get the number of interaction records.

        Returns:
            int: Number of interaction records.
        """
        return len(self.inter_feat)

    @property
    def avg_actions_of_users(self):
        """Get the average number of users' interaction records.

        Returns:
            numpy.float64: Average number of users' interaction records.
        """
        return np.mean(list(Counter(self.inter_feat[self.uid_field].numpy()).values()))

    @property
    def avg_actions_of_items(self):
        """Get the average number of items' interaction records.

        Returns:
            numpy.float64: Average number of items' interaction records.
        """
        return np.mean(list(Counter(self.inter_feat[self.iid_field].numpy()).values()))

    @property
    def sparsity(self):
        """Get the sparsity of this dataset.

        Returns:
            float: Sparsity of this dataset.
        """
        return 1 - self.inter_num / self.user_num / self.item_num

    def _check_field(self, *field_names):
        """Given a name of attribute, check if it's exist.

        Args:
            *field_names (str): Fields to be checked.
        """
        for field_name in field_names:
            if getattr(self, field_name, None) is None:
                raise ValueError(f'{field_name} isn\'t set.')

    @dlapi.set()
    def join(self, df):
        """Given interaction feature, join user/item feature into it.

        Args:
            df (Interaction): Interaction feature to be joint.

        Returns:
            Interaction: Interaction feature after joining operation.
        """
        if self.user_feat is not None and self.uid_field in df:
            df.update(self.user_feat[df[self.uid_field]])
        if self.item_feat is not None and self.iid_field in df:
            #print("join@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            df.update(self.item_feat[df[self.iid_field]])
        return df

    def __getitem__(self, index, join=True):
        df = self.inter_feat[index]
        return self.join(df) if join else df

    def __len__(self):
        return len(self.inter_feat)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [self.dataset_name]
        if self.uid_field:
            info.extend([
                f'The number of users: {self.user_num}', f'Average actions of users: {self.avg_actions_of_users}'
            ])
        if self.iid_field:
            info.extend([
                f'The number of items: {self.item_num}', f'Average actions of items: {self.avg_actions_of_items}'
            ])
        info.append(f'The number of inters: {self.inter_num}')
        if self.uid_field and self.iid_field:
            info.append(f'The sparsity of the dataset: {self.sparsity * 100}%')
        info.append(f'Remain Fields: {list(self.field2type)}')
        return '\n'.join(info)

    def copy(self, new_inter_feat):
        """Given a new interaction feature, return a new :class:`Dataset` object,
        whose interaction feature is updated with ``new_inter_feat``, and all the other attributes the same.

        Args:
            new_inter_feat (Interaction): The new interaction feature need to be updated.

        Returns:
            :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
        """
        nxt = copy.copy(self)
        nxt.inter_feat = new_inter_feat
        return nxt

    def _drop_unused_col(self):
        """Drop columns which are loaded for data preparation but not used in model.
        """
        unused_col = self.config['unused_col']
        if unused_col is None:
            return

        for feat_name, unused_fields in unused_col.items():
            feat = getattr(self, feat_name + '_feat')
            for field in unused_fields:
                if field not in feat:
                    self.logger.warning(
                        f'Field [{field}] is not in [{feat_name}_feat], which can not be set in `unused_col`.'
                    )
                    continue
                self._del_col(feat, field)

    def _grouped_index(self, group_by_list):
        index = {}
        for i, key in enumerate(group_by_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        #print("index@@@@@@@@:", len(index.keys()))
        return index.keys(), index.values()

    def _calcu_split_ids(self, tot, ratios):
        """Given split ratios, and total number, calculate the number of each part after splitting.

        Other than the first one, each part is rounded down.

        Args:
            tot (int): Total number.
            ratios (list): List of split ratios. No need to be normalized.

        Returns:
            list: Number of each part after splitting.
        """
        cnt = [int(ratios[i] * tot) for i in range(len(ratios))]
        cnt[0] = tot - sum(cnt[1:])
        split_ids = np.cumsum(cnt)[:-1]
        return list(split_ids)

    def _calcu_split_ids_for_pop(self, uid, group_index, ratios):
        """Given split ratios, and total number, calculate the number of each part after splitting.

        Other than the first one, each part is rounded down.

        Args:
            tot (int): Total number.
            ratios (list): List of split ratios. No need to be normalized.

        Returns:
            list: Number of each part after splitting.
        """
        tot = len(group_index)
        cnt = [int(ratios[i] * tot) for i in range(len(ratios))]
        cnt[0] = tot - sum(cnt[1:])

        train_index, valid_index, test_index = [], [], []
        valid_unpop = int(cnt[1] / 2)
        valid_pop = cnt[1] - valid_unpop
        test_unpop = int(cnt[2] / 2)
        test_pop = cnt[2] - test_unpop
        #print("!!!!!!!!!!!!!!!!!", cnt[0], cnt[1], cnt[2])
        #print("@@@@@@@@@@@@@@@",valid_pop, valid_unpop, test_pop, test_unpop)

        item_list = self.inter_feat[self.iid_field].numpy()
        #print("pos:", group_index)
        test_pop_cnt, test_unpop_cnt, valid_pop_cnt, valid_unpop_cnt, train_pop_cnt, train_unpop_cnt = 0, 0, 0, 0, 0, 0
        test_cnt,  valid_cnt, train_cnt = 0, 0, 0
        unpop_cnt = 0
        thresh = self.config['thresh']
        for i in range(tot - 1, -1, -1):
            item = item_list[group_index[i]]
            if self.item_inter_num[item] > thresh:
                if valid_pop_cnt < valid_pop:
                    valid_index.append(group_index[i])
                    valid_pop_cnt += 1
                    valid_cnt += 1
                elif valid_pop_cnt >= valid_pop and test_pop_cnt < test_pop:
                    test_index.append(group_index[i])
                    test_pop_cnt += 1
                    test_cnt += 1
                elif test_pop_cnt >= test_pop and valid_pop_cnt >= valid_pop:
                    train_index.append(group_index[i])
                    train_pop_cnt += 1
                    train_cnt += 1
            else:
                unpop_cnt += 1
                if valid_unpop_cnt < valid_unpop and (unpop_cnt % 2) == 1:
                    valid_index.append(group_index[i])
                    valid_unpop_cnt += 1
                    valid_cnt += 1
                #elif valid_unpop_cnt >= valid_unpop and test_unpop_cnt < test_unpop:
                elif test_unpop_cnt < test_unpop and (unpop_cnt % 2) == 0:
                    test_index.append(group_index[i])
                    test_unpop_cnt += 1
                    test_cnt += 1
                elif test_unpop_cnt >= test_unpop and valid_unpop_cnt >= valid_unpop:
                    train_index.append(group_index[i])
                    train_unpop_cnt += 1
                    train_cnt += 1

        if test_unpop_cnt < test_unpop:
            cha = test_unpop - test_unpop_cnt
            for i in range(cha):
                tmp = train_index.pop()
                test_index.append(tmp)
                test_pop_cnt += 1
                train_pop_cnt -= 1
        if valid_unpop_cnt < valid_unpop:
            cha = valid_unpop - valid_unpop_cnt
            for i in range(cha):
                tmp = train_index.pop()
                valid_index.append(tmp)
                valid_pop_cnt += 1
                train_pop_cnt -= 1

        #print("%%%%%%%%%%%%%%%%%%%", train_cnt, valid_cnt, test_cnt)
        #print("&&&&&&&&&&&&&&&&&&&&", train_pop_cnt, train_unpop_cnt, valid_pop_cnt, valid_unpop_cnt, test_pop_cnt, test_unpop_cnt)

        result = []
        result.append(train_index)
        result.append(valid_index)
        result.append(test_index)

        #item_list = self.inter_feat[self.iid_field].numpy()
        #if uid == 103:
        #    print("all_item11111111111:", item_list[group_index])
        #    print("train_item111111111:", set(item_list[train_index]))
        #    print("valid_item111111111:", set(item_list[valid_index]))
        #    print("test_item111111111:", set(item_list[test_index]))
        return result

    def _calcu_split_ids_for_position(self, uid, group_index, ratios):
        train_index, valid_index, test_index = [], [], []
        item_list = self.inter_feat[self.iid_field].numpy()
        positions = self.inter_feat['position'].numpy()
        # print("pos:", group_index)
        tot = len(group_index)
        test_cnt, valid_cnt, train_cnt = 0, 0, 0
        cnt = 0
        for i in range(tot - 1, -1, -1):
            #item = item_list[group_index[i]]
            pos = positions[group_index[i]]
            #print("pos:", pos)
            if pos < 26:
                train_index.append(group_index[i])
                train_cnt += 1
            else:
                if (cnt % 2) == 1:
                    valid_index.append(group_index[i])
                    valid_cnt += 1
                    cnt += 1
                elif(cnt % 2) == 0:
                    test_index.append(group_index[i])
                    test_cnt += 1
                    cnt += 1

        #print("%%%%%%%%%%%%%%%%%%%", train_cnt, valid_cnt, test_cnt)
        #print("%%%%%%%%%%%%%%%%%%%", train_cnt, valid_cnt, test_cnt)
        #print("&&&&&&&&&&&&&&&&&&&&", train_pop_cnt, train_unpop_cnt, valid_pop_cnt, valid_unpop_cnt, test_pop_cnt, test_unpop_cnt)

        result = []
        result.append(train_index)
        result.append(valid_index)
        result.append(test_index)
        return result

    def _calcu_split_ids_for_genre(self, uid, group_index, ratios):
        g_cnt = [0] * 9
        tot = len(group_index)

        train_index, valid_index, test_index = [], [], []
        train_cnt, valid_cnt, test_cnt = 0, 0, 0
        genres = self.inter_feat['new_genre'].numpy()

        g_used = [0] * tot
        genre_class_list = [[] for i in range(8)]
        for id in range(tot - 1, -1, -1):
            genre = genres[group_index[id]]
            for g in genre:
                if g == 0:
                    break
                g_cnt[g] += 1
                genre_class_list[g - 1].append(id)

        genre_class_list.sort(key=len)
        for genre_list in genre_class_list:
            #print(genre_list)
            valid_gr_cnt, test_gr_cnt = 0, 0
            if len(genre_list) == 0:
                continue
            if len(genre_list) == 1:
                gr = genre_list[0]
                #print("gr_111111:", gr)
                if g_used[gr] == 0:
                    valid_index.append(group_index[gr])
                    valid_gr_cnt += 1
                    valid_cnt += 1
                    test_index.append(group_index[gr])
                    test_gr_cnt += 1
                    test_cnt += 1
                    g_used[gr] += 1

            elif len(genre_list) >= 2:
                for gr in genre_list:
                    if g_used[gr] > 0:
                        continue
                    if valid_gr_cnt == 0:
                        valid_index.append(group_index[gr])
                        valid_cnt += 1
                        #print("valid_cnt:", valid_cnt)
                        valid_gr_cnt += 1
                        g_used[gr] += 1
                        continue
                    if test_gr_cnt == 0:
                        test_index.append(group_index[gr])
                        test_cnt += 1
                        test_gr_cnt += 1
                        g_used[gr] += 1
                        continue
                    train_index.append(group_index[gr])
                    train_cnt += 1
                    g_used[gr] += 1

        #print("tot:", tot)
        #print("train_cnt:", train_cnt)
        #print("valid_cnt:", valid_cnt)
        #print("test_cnt:", test_cnt)

        #print("g_cnt:", g_cnt)
        self.user_genre[uid] = g_cnt
        result = []
        result.append(train_index)
        result.append(valid_index)
        result.append(test_index)
        return result

    def _calcu_split_ids_for_dr(self, uid, group_index, ratios):
        train_index, valid_index, test_index = [], [], []
        item_list = self.inter_feat[self.iid_field].numpy()
        positions = self.inter_feat['position'].numpy()
        # print("pos:", group_index)
        tot = len(group_index)
        test_cnt, valid_cnt, train_cnt = 0, 0, 0
        cnt = 0
        for i in range(tot - 1, -1, -1):
            #item = item_list[group_index[i]]
            pos = positions[group_index[i]]
            #print("pos:", pos)
            if pos < 26:
                train_index.append(group_index[i])
                train_cnt += 1
            else:
                if (cnt % 5) == 0:
                    train_index.append(group_index[i])
                    train_cnt += 1
                    cnt += 1
                elif (cnt % 5) == 1 or (cnt % 5) == 2:
                    valid_index.append(group_index[i])
                    valid_cnt += 1
                    cnt += 1
                elif (cnt % 5) == 3 or (cnt % 5) == 4:
                    test_index.append(group_index[i])
                    test_cnt += 1
                    cnt += 1

        result = []
        result.append(train_index)
        result.append(valid_index)
        result.append(test_index)
        return result

    def tj_pop(self, index_list):
        [train_ids, valid_ids, test_ids] = index_list
        item_list = self.inter_feat[self.iid_field].numpy()
        #print("item_list:", item_list)
        thresh = self.config['thresh']
        test_pop_cnt, test_unpop_cnt, valid_pop_cnt, valid_unpop_cnt, train_pop_cnt, train_unpop_cnt = 0, 0, 0, 0, 0, 0
        for id in train_ids:
            item = item_list[id]
            if self.item_inter_num[item] > thresh:
                train_pop_cnt += 1
            else:
                train_unpop_cnt += 1

        for id in valid_ids:
            item = item_list[id]
            if self.item_inter_num[item] > thresh:
                valid_pop_cnt += 1
            else:
                valid_unpop_cnt += 1

        for id in test_ids:
            item = item_list[id]
            if self.item_inter_num[item] > thresh:
                test_pop_cnt += 1
            else:
                test_unpop_cnt += 1

        result = [train_pop_cnt + train_unpop_cnt, valid_pop_cnt + valid_unpop_cnt, test_pop_cnt + test_unpop_cnt,
                  train_pop_cnt, train_unpop_cnt, valid_pop_cnt, valid_unpop_cnt, test_pop_cnt, test_unpop_cnt]
        return result

    def split_by_ratio(self, ratios, group_by=None):
        """Split interaction records by ratios.

        Args:
            ratios (list): List of split ratios. No need to be normalized.
            group_by (str, optional): Field name that interaction records should grouped by before splitting.
                Defaults to ``None``

        Returns:
            list: List of :class:`~Dataset`, whose interaction features has been split.

        Note:
            Other than the first one, each part is rounded down.
        """
        self.logger.debug(f'split by ratios [{ratios}], group_by=[{group_by}]')
        tot_ratio = sum(ratios)
        ratios = [_ / tot_ratio for _ in ratios]

        if group_by is None:
            tot_cnt = self.__len__()
            split_ids = self._calcu_split_ids(tot=tot_cnt, ratios=ratios)
            next_index = [range(start, end) for start, end in zip([0] + split_ids, split_ids + [tot_cnt])]
        else:
            grouped_user_list, grouped_inter_feat_index = self._grouped_index(self.inter_feat[group_by].numpy())
            next_index = [[] for _ in range(len(ratios))]
            for uid, grouped_index in zip(grouped_user_list, grouped_inter_feat_index):
                tot_cnt = len(grouped_index)
                if self.config['split_mode'] == 0:
                    split_ids = self._calcu_split_ids(tot=tot_cnt, ratios=ratios)

                    for index, start, end in zip(next_index, [0] + split_ids, split_ids + [tot_cnt]):
                        index.extend(grouped_index[start:end])

                elif self.config['split_mode'] == 1:  # for popularity
                    pop_split_ids = self._calcu_split_ids_for_pop(uid= uid, group_index=grouped_index, ratios=ratios)
                    for index, ids in zip(next_index, pop_split_ids):
                        index.extend(ids)

                elif self.config['split_mode'] == 2: # for position
                    position_split_ids = self._calcu_split_ids_for_position(uid= uid, group_index=grouped_index, ratios=ratios)
                    for index, ids in zip(next_index, position_split_ids):
                        index.extend(ids)

                elif self.config['split_mode'] == 3:
                    position_split_ids = self._calcu_split_ids_for_dr(uid=uid, group_index=grouped_index,
                                                                            ratios=ratios)
                    for index, ids in zip(next_index, position_split_ids):
                        index.extend(ids)

                elif self.config['split_mode'] == 4:
                    position_split_ids = self._calcu_split_ids_for_genre(uid=uid, group_index=grouped_index,
                                                                            ratios=ratios)
                    for index, ids in zip(next_index, position_split_ids):
                        index.extend(ids)

        self._drop_unused_col()
        #print("*********self.inter_feat:", self.inter_feat['genre'])
        next_df = [self.inter_feat[index] for index in next_index]
        #print("!!!!!!!!!", next_df[0]['genre'])
        next_ds = [self.copy(_) for _ in next_df]
        #print("train@@@@@@@@@@@@@@", next_ds[0][:10])

        return next_ds

    def _split_index_by_leave_one_out(self, grouped_index, leave_one_num):
        """Split indexes by strategy leave one out.

        Args:
            grouped_index (list of list of int): Index to be split.
            leave_one_num (int): Number of parts whose length is expected to be ``1``.

        Returns:
            list: List of index that has been split.
        """
        next_index = [[] for _ in range(leave_one_num + 1)]
        for index in grouped_index:
            index = list(index)
            tot_cnt = len(index)
            legal_leave_one_num = min(leave_one_num, tot_cnt - 1)
            pr = tot_cnt - legal_leave_one_num
            next_index[0].extend(index[:pr])
            for i in range(legal_leave_one_num):
                next_index[-legal_leave_one_num + i].append(index[pr])
                pr += 1
        return next_index

    def leave_one_out(self, group_by, leave_one_num=1):
        """Split interaction records by leave one out strategy.

        Args:
            group_by (str): Field name that interaction records should grouped by before splitting.
            leave_one_num (int, optional): Number of parts whose length is expected to be ``1``.
                Defaults to ``1``.

        Returns:
            list: List of :class:`~Dataset`, whose interaction features has been split.
        """
        self.logger.debug(f'leave one out, group_by=[{group_by}], leave_one_num=[{leave_one_num}]')
        if group_by is None:
            raise ValueError('leave one out strategy require a group field')

        grouped_inter_feat_index = self._grouped_index(self.inter_feat[group_by].numpy())
        next_index = self._split_index_by_leave_one_out(grouped_inter_feat_index, leave_one_num)

        self._drop_unused_col()
        next_df = [self.inter_feat[index] for index in next_index]
        next_ds = [self.copy(_) for _ in next_df]
        return next_ds

    def shuffle(self):
        """Shuffle the interaction records inplace.
        """
        self.inter_feat.shuffle()

    def sort(self, by, ascending=True):
        """Sort the interaction records inplace.

        Args:
            by (str or list of str): Field that as the key in the sorting process.
            ascending (bool or list of bool, optional): Results are ascending if ``True``, otherwise descending.
                Defaults to ``True``
        """
        self.inter_feat.sort(by=by, ascending=ascending)

    def build(self, eval_setting):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Args:
            eval_setting (:class:`~recbole.config.eval_setting.EvalSetting`):
                Object contains evaluation settings, which guide the data processing procedure.

        Returns:
            list: List of built :class:`Dataset`.
        """
        if self.benchmark_filename_list is not None:
            cumsum = list(np.cumsum(self.file_size_list))
            datasets = [self.copy(self.inter_feat[start:end]) for start, end in zip([0] + cumsum[:-1], cumsum)]
            return datasets

        ordering_args = eval_setting.ordering_args
        if ordering_args['strategy'] == 'shuffle':
            self.shuffle()
        elif ordering_args['strategy'] == 'by':
            self.sort(by=ordering_args['field'], ascending=ordering_args['ascending'])

        group_field = eval_setting.group_field

        split_args = eval_setting.split_args
        if split_args['strategy'] == 'by_ratio':
            datasets = self.split_by_ratio(split_args['ratios'], group_by=group_field)
        elif split_args['strategy'] == 'by_value':
            raise NotImplementedError()
        elif split_args['strategy'] == 'loo':
            datasets = self.leave_one_out(group_by=group_field, leave_one_num=split_args['leave_one_num'])
        else:
            datasets = self

        return datasets

    def save(self, filepath):
        """Saving this :class:`Dataset` object to local path.

        Args:
            filepath (str): path of saved dir.
        """
        if (filepath is None) or (not os.path.isdir(filepath)):
            raise ValueError(f'Filepath [{filepath}] need to be a dir.')

        self.logger.debug(f'Saving into [{filepath}]')
        basic_info = {
            'field2type': self.field2type,
            'field2source': self.field2source,
            'field2id_token': self.field2id_token,
            'field2seqlen': self.field2seqlen
        }

        with open(os.path.join(filepath, 'basic-info.json'), 'w', encoding='utf-8') as file:
            json.dump(basic_info, file)

        feats = ['inter', 'user', 'item']
        for name in feats:
            df = getattr(self, f'{name}_feat')
            if df is not None:
                df.to_csv(os.path.join(filepath, f'{name}.csv'))

    @dlapi.set()
    def get_user_feature(self):
        """
        Returns:
            Interaction: user features
        """
        if self.user_feat is None:
            self._check_field('uid_field')
            return Interaction({self.uid_field: torch.arange(self.user_num)})
        else:
            return self.user_feat

    @dlapi.set()
    def get_item_feature(self):
        """
        Returns:
            Interaction: item features
        """
        if self.item_feat is None:
            self._check_field('iid_field')
            return Interaction({self.iid_field: torch.arange(self.item_num)})
        else:
            return self.item_feat

    def _create_sparse_matrix(self, df_feat, source_field, target_field, form='coo', value_field=None):
        """Get sparse matrix that describe relations between two fields.

        Source and target should be token-like fields.

        Sparse matrix has shape (``self.num(source_field)``, ``self.num(target_field)``).

        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = df_feat[value_field][src, tgt]``.

        Args:
            df_feat (Interaction): Feature where src and tgt exist.
            source_field (str): Source field
            target_field (str): Target field
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        src = df_feat[source_field]
        tgt = df_feat[target_field]
        if value_field is None:
            data = np.ones(len(df_feat))
        else:
            if value_field not in df_feat:
                raise ValueError(f'Value_field [{value_field}] should be one of `df_feat`\'s features.')
            data = df_feat[value_field]
        mat = coo_matrix((data, (src, tgt)), shape=(self.num(source_field), self.num(target_field)))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError(f'Sparse matrix format [{form}] has not been implemented.')

    def _create_graph(self, tensor_feat, source_field, target_field, form='dgl', value_field=None):
        """Get graph that describe relations between two fields.

        Source and target should be token-like fields.

        For an edge of <src, tgt>, ``graph[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``graph[src, tgt] = df_feat[value_field][src, tgt]``.

        Currently, we support graph in `DGL`_ and `PyG`_.

        Args:
            tensor_feat (Interaction): Feature where src and tgt exist.
            source_field (str): Source field
            target_field (str): Target field
            form (str, optional): Library of graph data structure. Defaults to ``dgl``.
            value_field (str, optional): edge attributes of graph, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            Graph of relations.

        .. _DGL:
            https://www.dgl.ai/

        .. _PyG:
            https://github.com/rusty1s/pytorch_geometric
        """
        src = tensor_feat[source_field]
        tgt = tensor_feat[target_field]

        if form == 'dgl':
            import dgl
            graph = dgl.graph((src, tgt))
            if value_field is not None:
                if isinstance(value_field, str):
                    value_field = {value_field}
                for k in value_field:
                    graph.edata[k] = tensor_feat[k]
            return graph
        elif form == 'pyg':
            from torch_geometric.data import Data
            edge_attr = tensor_feat[value_field] if value_field else None
            graph = Data(edge_index=torch.stack([src, tgt]), edge_attr=edge_attr)
            return graph
        else:
            raise NotImplementedError(f'Graph format [{form}] has not been implemented.')

    @dlapi.set()
    def inter_matrix(self, form='coo', value_field=None):
        """Get sparse matrix that describe interactions between user_id and item_id.

        Sparse matrix has shape (user_num, item_num).

        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = self.inter_feat[src, tgt]``.

        Args:
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if not self.uid_field or not self.iid_field:
            raise ValueError('dataset does not exist uid/iid, thus can not converted to sparse matrix.')
        return self._create_sparse_matrix(self.inter_feat, self.uid_field, self.iid_field, form, value_field)

    def _history_matrix(self, row, value_field=None):
        """Get dense matrix describe user/item's history interaction records.

        ``history_matrix[i]`` represents ``i``'s history interacted item_id.

        ``history_value[i]`` represents ``i``'s history interaction records' values.
            ``0`` if ``value_field = None``.

        ``history_len[i]`` represents number of ``i``'s history interaction records.

        ``0`` is used as padding.

        Args:
            row (str): ``user`` or ``item``.
            value_field (str, optional): Data of matrix, which should exist in ``self.inter_feat``.
                Defaults to ``None``.

        Returns:
            tuple:
                - History matrix (torch.Tensor): ``history_matrix`` described above.
                - History values matrix (torch.Tensor): ``history_value`` described above.
                - History length matrix (torch.Tensor): ``history_len`` described above.
        """
        self._check_field('uid_field', 'iid_field')

        user_ids, item_ids = self.inter_feat[self.uid_field].numpy(), self.inter_feat[self.iid_field].numpy()
        if value_field is None:
            values = np.ones(len(self.inter_feat))
        else:
            if value_field not in self.inter_feat:
                raise ValueError(f'Value_field [{value_field}] should be one of `inter_feat`\'s features.')
            values = self.inter_feat[value_field].numpy()

        if row == 'user':
            row_num, max_col_num = self.user_num, self.item_num
            row_ids, col_ids = user_ids, item_ids
        else:
            row_num, max_col_num = self.item_num, self.user_num
            row_ids, col_ids = item_ids, user_ids

        history_len = np.zeros(row_num, dtype=np.int64)
        for row_id in row_ids:
            history_len[row_id] += 1

        col_num = np.max(history_len)
        if col_num > max_col_num * 0.2:
            self.logger.warning(
                f'Max value of {row}\'s history interaction records has reached '
                f'{col_num / max_col_num * 100}% of the total.'
            )

        history_matrix = np.zeros((row_num, col_num), dtype=np.int64)
        history_value = np.zeros((row_num, col_num))
        history_len[:] = 0
        for row_id, value, col_id in zip(row_ids, values, col_ids):
            history_matrix[row_id, history_len[row_id]] = col_id
            history_value[row_id, history_len[row_id]] = value
            history_len[row_id] += 1

        return torch.LongTensor(history_matrix), torch.FloatTensor(history_value), torch.LongTensor(history_len)

    @dlapi.set()
    def history_item_matrix(self, value_field=None):
        """Get dense matrix describe user's history interaction records.

        ``history_matrix[i]`` represents user ``i``'s history interacted item_id.

        ``history_value[i]`` represents user ``i``'s history interaction records' values,
        ``0`` if ``value_field = None``.

        ``history_len[i]`` represents number of user ``i``'s history interaction records.

        ``0`` is used as padding.

        Args:
            value_field (str, optional): Data of matrix, which should exist in ``self.inter_feat``.
                Defaults to ``None``.

        Returns:
            tuple:
                - History matrix (torch.Tensor): ``history_matrix`` described above.
                - History values matrix (torch.Tensor): ``history_value`` described above.
                - History length matrix (torch.Tensor): ``history_len`` described above.
        """
        return self._history_matrix(row='user', value_field=value_field)

    @dlapi.set()
    def history_user_matrix(self, value_field=None):
        """Get dense matrix describe item's history interaction records.

        ``history_matrix[i]`` represents item ``i``'s history interacted item_id.

        ``history_value[i]`` represents item ``i``'s history interaction records' values,
        ``0`` if ``value_field = None``.

        ``history_len[i]`` represents number of item ``i``'s history interaction records.

        ``0`` is used as padding.

        Args:
            value_field (str, optional): Data of matrix, which should exist in ``self.inter_feat``.
                Defaults to ``None``.

        Returns:
            tuple:
                - History matrix (torch.Tensor): ``history_matrix`` described above.
                - History values matrix (torch.Tensor): ``history_value`` described above.
                - History length matrix (torch.Tensor): ``history_len`` described above.
        """
        return self._history_matrix(row='item', value_field=value_field)

    @dlapi.set()
    def get_preload_weight(self, field):
        """Get preloaded weight matrix, whose rows are sorted by token ids.

        ``0`` is used as padding.

        Args:
            field (str): preloaded feature field name.

        Returns:
            numpy.ndarray: preloaded weight matrix. See :doc:`../user_guide/data/data_args` for details.
        """
        if field not in self._preloaded_weight:
            raise ValueError(f'Field [{field}] not in preload_weight')
        return self._preloaded_weight[field]

    def _dataframe_to_interaction(self, data):
        """Convert :class:`pandas.DataFrame` to :class:`~recbole.data.interaction.Interaction`.

        Args:
            data (pandas.DataFrame): data to be converted.

        Returns:
            :class:`~recbole.data.interaction.Interaction`: Converted data.
        """
        new_data = {}
        for k in data:
            ftype = self.field2type[k]

            if ftype == FeatureType.TOKEN:
                value = data[k].values.astype(float)
                new_data[k] = torch.LongTensor(value)
            elif ftype == FeatureType.FLOAT:
                value = data[k].values.astype(float)
                new_data[k] = torch.FloatTensor(value)
            elif ftype == FeatureType.TOKEN_SEQ:
                value = data[k].values
                seq_data = [torch.LongTensor(d[:self.field2seqlen[k]]) for d in value]
                new_data[k] = rnn_utils.pad_sequence(seq_data, batch_first=True)
            elif ftype == FeatureType.FLOAT_SEQ:
                value = data[k].values
                seq_data = [torch.FloatTensor(d[:self.field2seqlen[k]]) for d in value]
                new_data[k] = rnn_utils.pad_sequence(seq_data, batch_first=True)
        return Interaction(new_data)
