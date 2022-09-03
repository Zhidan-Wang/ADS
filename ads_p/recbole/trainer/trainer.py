# @Time   : 2020/6/26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/7, 2020/9/26, 2020/9/26, 2020/10/01, 2020/9/16, 2020/10/8, 2020/10/15, 2020/11/20
# @Author : Zihan Lin, Yupeng Hou, Yushuo Chen, Shanlei Mu, Xingyu Pan, Hui Wang, Xinyan Fan, Chen Yang
# @Email  : linzihan.super@foxmail.com, houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, slmu@ruc.edu.cn, panxy@ruc.edu.cn, hui.wang@ruc.edu.cn, xinyan.fan@ruc.edu.cn, 254170321@qq.com

r"""
recbole.trainer.trainer
################################
"""

import os
from logging import getLogger
from time import time

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

from recbole.data.interaction import Interaction
from recbole.evaluator import ProxyEvaluator
from recbole.utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str, \
    DataLoaderType, KGDataLoaderState


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.model_name = config['model']

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
    resume_checkpoint() and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model, item_popularity):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.checkpoint_dir = config['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        saved_model_file = '{}-{}.pth'.format(self.config['model'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
        self.weight_decay = config['weight_decay']

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -1
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        self.eval_type = config['eval_type']
        self.evaluator = ProxyEvaluator(config)
        self.item_tensor = None
        self.tot_item_num = None

        self.item_popularity = item_popularity

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(self.model.parameters(), lr=self.learning_rate)
            if self.weight_decay > 0:
                self.logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        #print("!!!!!!!!!!!!model embedding before:", self.model.item_embedding.weight)
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                enumerate(train_data),
                total=len(train_data),
                desc=f"Train {epoch_idx:>5}",
            ) if show_progress else enumerate(train_data)
        )
        for batch_idx, interaction in iter_data:
            interaction = interaction.to(self.device)

            #for parm in self.model.parameters():
            #    parm.data.clamp_(-0.001, 0.001)
            #print("!!!!!!!!!!!!model embedding before:", self.model.item_embedding.weight)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            #print("loss:", loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            #print("!!!!!!!!!!!!model embedding after:", self.model.item_embedding.weight)
        return total_loss

    def _valid_epoch(self, valid_data, show_progress=False):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        return valid_score, valid_result

    def _save_checkpoint(self, epoch):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, self.saved_model_file)

    def resume_checkpoint(self, resume_file):
        r"""Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file

        """
        resume_file = str(resume_file)
        checkpoint = torch.load(resume_file)
        self.start_epoch = checkpoint['epoch'] + 1
        self.cur_step = checkpoint['cur_step']
        self.best_valid_score = checkpoint['best_valid_score']

        # load architecture params from checkpoint
        if checkpoint['config']['model'].lower() != self.config['model'].lower():
            self.logger.warning(
                'Architecture configuration given in config file is different from that of checkpoint. '
                'This may yield an exception while state_dict is being loaded.'
            )
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        self.logger.info(message_output)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = self.config['loss_decimal_place'] or 4
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = 'train_loss%d: %.' + str(des) + 'f'
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += 'train loss:' + des % losses
        return train_loss_output + ']'

    def fit_dis(self, train_data, valid_data, test_data, verbose=True, saved=True, show_progress=False, callback_fn=None):

        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1)

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = 'Saving current: %s' % self.saved_model_file
                    if verbose:
                        self.logger.info(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score = self.dis_evaluate(valid_data, epoch=epoch_idx, writer=writer, load_best_model=False, show_progress=show_progress)
                valid_result = valid_score

                test_result = self.dis_evaluate(test_data, epoch=epoch_idx, writer=writer, load_best_model=False, show_progress=show_progress)
                test_score = test_result

                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )

                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                test_score_output = "epoch %d testing [time: %.2fs, test_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, test_score)
                #valid_result_output = 'valid result: \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(test_score_output)
                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = 'Saving current best: %s' % self.saved_model_file
                        if verbose:
                            self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                    (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        return self.best_valid_score, self.best_valid_result

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1)

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = 'Saving current: %s' % self.saved_model_file
                    if verbose:
                        self.logger.info(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)

                #valid_pop_result, valid_pop_num, valid_pop_mean = self.evaluate_pop(valid_data)

                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    #self.logger.info(valid_pop_result)
                    #self.logger.info(valid_pop_num)
                    #self.logger.info(valid_pop_mean)

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = 'Saving current best: %s' % self.saved_model_file
                        if verbose:
                            self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        return self.best_valid_score, self.best_valid_result

    def _full_sort_batch_eval(self, batched_data):
        #interaction, history_index, swap_row, swap_col_after, swap_col_before = batched_data
        interaction, history_index, swap_row1, swap_col_after1, swap_col_before1,swap_row2, swap_col_after2, swap_col_before2 = batched_data
        try:
            # Note: interaction without item ids
            #print("self.model:", self.model_name)
            if self.model_name == 'BPR_PC':
                scores = self.model.full_sort_predict(interaction.to(self.device), self.item_popularity)
            else:
                scores = self.model.full_sort_predict(interaction.to(self.device))
        except NotImplementedError:
            new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(self.item_tensor[:batch_size])
            if batch_size <= self.test_batch_size:
                scores = self.model.predict(new_inter)

            else:
                scores = self._spilt_predict(new_inter, batch_size)

        scores = scores.view(-1, self.tot_item_num)
        #print("score!!!!!!!", scores)
        #print("score!!", scores.size())
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf

        swap_row1 = swap_row1.to(self.device)
        swap_col_after1 = swap_col_after1.to(self.device)
        swap_col_before1 = swap_col_before1.to(self.device)

        swap_row2 = swap_row2.to(self.device)
        swap_col_after2 = swap_col_after2.to(self.device)
        swap_col_before2 = swap_col_before2.to(self.device)
        #print("!!!!!!!!!!", swap_col_before)
        tmp_scores = scores

        scores[swap_row1, swap_col_after1] = tmp_scores[swap_row1, swap_col_before1]
        scores[swap_row2, swap_col_after2] = tmp_scores[swap_row2, swap_col_before2]
        #print("after score", scores)
        return interaction, scores

    def _full_sort_batch_eval_pop(self, batched_data):
        interaction, history_index, swap_row1, swap_col_after1, swap_col_before1,swap_row2, swap_col_after2, swap_col_before2 = batched_data

        if self.model_name == 'BPR_PC':
            scores = self.model.full_sort_predict(interaction.to(self.device), self.item_popularity)
        else:
            scores = self.model.full_sort_predict(interaction.to(self.device))

        scores = scores.view(-1, self.tot_item_num)
        #print("score!!!!!!!", scores)
        #print("score!!", scores.size())
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf

        return interaction, scores
    @torch.no_grad()
    def dis_evaluate(self, eval_data, emb_model, thresh, load_best_model=True, model_file=None, show_progress=False):
        if not eval_data:
            return

        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        iter_data = (
            tqdm(
                enumerate(eval_data),
                total=len(eval_data),
                desc=f"Evaluate   ",
            ) if show_progress else enumerate(eval_data)
        )

        #pos_tot = 0
        #pos_true_tot = 0
        result = 0
        for batch_idx, batched_data in iter_data:
            if eval_data.dl_type == DataLoaderType.FULL:
                print("full!!!!!!!!!!!!!!!")
                #interaction, scores = self._full_sort_batch_eval(batched_data)

            else:
                interaction = batched_data
                batch_size = interaction.length
                if batch_size <= self.test_batch_size:
                    #pos, pos_true = self.model.predict(interaction.to(self.device),writer)
                    interaction = interaction.to(self.device)
                    #print("interaction", interaction)
                    items = interaction['item_id']
                    threshold = torch.tensor(np.full(len(interaction['pop']), thresh))
                    pos_index = interaction['pop'] >= threshold
                    pop_label = torch.tensor(np.where(pos_index, 1, 0))
                    # print("pop_label:", pop_label)
                    enc_1 = emb_model.item_embedding(items)

                    scores = self.model.predict(enc_1, pop_label)
                else:
                    interaction = interaction.to(self.device)
                    items = interaction['item_id']
                    threshold = torch.tensor(np.full(len(interaction['pop']), 200))
                    pos_index = interaction['pop'] >= threshold
                    pop_label = torch.tensor(np.where(pos_index, 1, 0))
                    # print("pop_label:", pop_label)
                    enc_1 = emb_model.item_embedding(items)

                    scores = self._spilt_predict(enc_1, pop_label, batch_size)
                #pos_tot += pos
                #pos_true_tot += pos_true
                result += scores
        print("Result !!!!!!!!", result)

        #result = float(result) / len(eval_data)
        result = float(result) / 9531
        #result = float(pos_true_tot) / float(pos_tot)

        return result

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        if eval_data.dl_type == DataLoaderType.FULL:
            if self.item_tensor is None:
                self.item_tensor = eval_data.get_item_feature().to(self.device).repeat(eval_data.step)
            self.tot_item_num = eval_data.dataset.item_num
            #self.tot_item_num = 3417
        batch_matrix_list = []
        iter_data = (
            tqdm(
                enumerate(eval_data),
                total=len(eval_data),
                desc=f"Evaluate   ",
            ) if show_progress else enumerate(eval_data)
        )
        for batch_idx, batched_data in iter_data:
            #print("batched_data for eval!!!!!!!!", batched_data)
            if eval_data.dl_type == DataLoaderType.FULL:
                interaction, scores = self._full_sort_batch_eval(batched_data)
            else:
                interaction = batched_data
                batch_size = interaction.length
                if batch_size <= self.test_batch_size:
                    scores = self.model.predict(interaction.to(self.device))
                else:
                    scores = self._spilt_predict(interaction, batch_size)

            batch_matrix = self.evaluator.collect(interaction, scores)
            batch_matrix_list.append(batch_matrix)
        result = self.evaluator.evaluate(batch_matrix_list, eval_data)

        return result

    def write_rec_list(self, file_path, eval_data, user_mp, item_mp):
        self.topk = self.config['topk']
        iter_data = enumerate(eval_data)
        for batch_idx, batched_data in iter_data:
            interaction, scores = self._full_sort_batch_eval_pop(batched_data)
            _, topk_idx = torch.topk(scores, max(self.topk), dim=-1)
            with open(file_path, 'w') as fn:
                for u in range(topk_idx.shape[0]):
                    user = user_mp[u]
                    for p, i in enumerate(topk_idx[u]):
                        #print(i)
                        item = item_mp[i - 1]
                        position = p + 1
                        fn.write(str(user) + '\t' + str(item) + '\t' + str(position) + '\n')

    def evaluate_pop(self, eval_data, load_best_model=False, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        self.topk = self.config['topk']
        positive_items = eval_data.uid2positive
        #print("positive:", list(positive_items))

        if eval_data.dl_type == DataLoaderType.FULL:
            if self.item_tensor is None:
                self.item_tensor = eval_data.get_item_feature().to(self.device).repeat(eval_data.step)
            self.tot_item_num = eval_data.dataset.item_num

        iter_data = (
            tqdm(
                enumerate(eval_data),
                total=len(eval_data),
                desc=f"Evaluate   ",
            ) if show_progress else enumerate(eval_data)
        )

        rec_num_list = [0] * self.config['p_class']
        rec_pos_score = [0] * self.config['p_class']
        rec_pos_num = [0] * self.config['p_class']
        rec_mean_score = [0] * self.config['p_class']
        threshes = self.get_threshes(self.config['p_class'])
        #threshes = [self.config['thresh']]
        for batch_idx, batched_data in iter_data:
            interaction, scores = self._full_sort_batch_eval_pop(batched_data)

            _, topk_idx = torch.topk(scores, max(self.topk), dim=-1)
            #print("item_pop::::::", self.item_popularity)
            for i in range(topk_idx.shape[0]):
                for j in range(topk_idx.shape[1]):
                    item = topk_idx[i][j]
                    i_pop = self.item_popularity[topk_idx[i][j]]
                    for n in range(len(rec_num_list)):
                        min_pop = threshes[n - 1] if n > 0 else 0
                        max_pop = threshes[n] if n < len(rec_num_list) - 1 else 100000

                        if i_pop > min_pop and i_pop <= max_pop:
                            rec_num_list[n] += 1
                            if item in list(positive_items[i+1]):
                                rec_pos_score[n] += scores[i][item]
                                rec_pos_num[n] += 1
                            break
        #print("result:", rec_num_list)
        #print("pos_num:", rec_pos_num)
        #print("pos_score:", rec_pos_score)
        for i in range(len(rec_pos_num)):
            if rec_pos_num[i] == 0:
                rec_mean_score[i] = 0
            else:
                rec_mean_score[i] = rec_pos_score[i].detach().numpy() / rec_pos_num[i]
        #print("mean_score:", rec_mean_score)

        return rec_num_list, rec_pos_num, rec_mean_score

    def get_threshes(self, n_class):
        threshes = []
        item_p = sorted(self.item_popularity)
        t = (len(item_p) / n_class)
        for i in range(1, n_class):
            idx = int(i * t)
            threshes.append(item_p[idx])
        #print("thresh!!!!!!!!", threshes)
        return threshes

    def _spilt_predict(self, interaction, batch_size):
        spilt_interaction = dict()
        for key, tensor in interaction.interaction.items():
            spilt_interaction[key] = tensor.split(self.test_batch_size, dim=0)
        num_block = (batch_size + self.test_batch_size - 1) // self.test_batch_size
        result_list = []
        for i in range(num_block):
            current_interaction = dict()
            for key, spilt_tensor in spilt_interaction.items():
                current_interaction[key] = spilt_tensor[i]
            result = self.model.predict(Interaction(current_interaction).to(self.device))
            if len(result.shape) == 0:
                result = result.unsqueeze(0)
            result_list.append(result)
        return torch.cat(result_list, dim=0)

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): Whether to show this figure, default: True
            save_path (str, optional): The data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        import matplotlib.pyplot as plt
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)


class KGTrainer(Trainer):
    r"""KGTrainer is designed for Knowledge-aware recommendation methods. Some of these models need to train the
    recommendation related task and knowledge related task alternately.

    """

    def __init__(self, config, model):
        super(KGTrainer, self).__init__(config, model)

        self.train_rec_step = config['train_rec_step']
        self.train_kg_step = config['train_kg_step']

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        if self.train_rec_step is None or self.train_kg_step is None:
            interaction_state = KGDataLoaderState.RSKG
        elif epoch_idx % (self.train_rec_step + self.train_kg_step) < self.train_rec_step:
            interaction_state = KGDataLoaderState.RS
        else:
            interaction_state = KGDataLoaderState.KG
        train_data.set_mode(interaction_state)
        if interaction_state in [KGDataLoaderState.RSKG, KGDataLoaderState.RS]:
            return super()._train_epoch(train_data, epoch_idx, show_progress=show_progress)
        elif interaction_state in [KGDataLoaderState.KG]:
            return super()._train_epoch(
                train_data, epoch_idx, loss_func=self.model.calculate_kg_loss, show_progress=show_progress
            )
        return None


class KGATTrainer(Trainer):
    r"""KGATTrainer is designed for KGAT, which is a knowledge-aware recommendation method.

    """

    def __init__(self, config, model):
        super(KGATTrainer, self).__init__(config, model)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        # train rs
        train_data.set_mode(KGDataLoaderState.RS)
        rs_total_loss = super()._train_epoch(train_data, epoch_idx, show_progress=show_progress)

        # train kg
        train_data.set_mode(KGDataLoaderState.KG)
        kg_total_loss = super()._train_epoch(
            train_data, epoch_idx, loss_func=self.model.calculate_kg_loss, show_progress=show_progress
        )

        # update A
        self.model.eval()
        with torch.no_grad():
            self.model.update_attentive_A()

        return rs_total_loss, kg_total_loss


class S3RecTrainer(Trainer):
    r"""S3RecTrainer is designed for S3Rec, which is a self-supervised learning based sequential recommenders.
        It includes two training stages: pre-training ang fine-tuning.

        """

    def __init__(self, config, model):
        super(S3RecTrainer, self).__init__(config, model)

    def save_pretrained_model(self, epoch, saved_model_file):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id
            saved_model_file (str): file name for saved pretrained model

        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, saved_model_file)

    def pretrain(self, train_data, verbose=True, show_progress=False):

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)

            if (epoch_idx + 1) % self.config['save_step'] == 0:
                saved_model_file = os.path.join(
                    self.checkpoint_dir,
                    '{}-{}-{}.pth'.format(self.config['model'], self.config['dataset'], str(epoch_idx + 1))
                )
                self.save_pretrained_model(epoch_idx, saved_model_file)
                update_output = 'Saving current: %s' % saved_model_file
                if verbose:
                    self.logger.info(update_output)

        return self.best_valid_score, self.best_valid_result

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        if self.model.train_stage == 'pretrain':
            return self.pretrain(train_data, verbose, show_progress)
        elif self.model.train_stage == 'finetune':
            return super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)
        else:
            raise ValueError("Please make sure that the 'train_stage' is 'pretrain' or 'finetune' ")


class MKRTrainer(Trainer):
    r"""MKRTrainer is designed for MKR, which is a knowledge-aware recommendation method.

    """

    def __init__(self, config, model):
        super(MKRTrainer, self).__init__(config, model)
        self.kge_interval = config['kge_interval']

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        rs_total_loss, kg_total_loss = 0., 0.

        # train rs
        self.logger.info('Train RS')
        train_data.set_mode(KGDataLoaderState.RS)
        rs_total_loss = super()._train_epoch(
            train_data, epoch_idx, loss_func=self.model.calculate_rs_loss, show_progress=show_progress
        )

        # train kg
        if epoch_idx % self.kge_interval == 0:
            self.logger.info('Train KG')
            train_data.set_mode(KGDataLoaderState.KG)
            kg_total_loss = super()._train_epoch(
                train_data, epoch_idx, loss_func=self.model.calculate_kg_loss, show_progress=show_progress
            )

        return rs_total_loss, kg_total_loss


class TraditionalTrainer(Trainer):
    r"""TraditionalTrainer is designed for Traditional model(Pop,ItemKNN), which set the epoch to 1 whatever the config.

    """

    def __init__(self, config, model):
        super(TraditionalTrainer, self).__init__(config, model)
        self.epochs = 1  # Set the epoch to 1 when running memory based model


class xgboostTrainer(AbstractTrainer):
    """xgboostTrainer is designed for XGBOOST.

    """

    def __init__(self, config, model):
        super(xgboostTrainer, self).__init__(config, model)

        self.xgb = __import__('xgboost')

        self.logger = getLogger()
        self.label_field = config['LABEL_FIELD']
        self.xgb_model = config['xgb_model']
        self.convert_token_to_onehot = self.config['convert_token_to_onehot']

        # DMatrix params
        self.weight = config['xgb_weight']
        self.base_margin = config['xgb_base_margin']
        self.missing = config['xgb_missing']
        self.silent = config['xgb_silent']
        self.feature_names = config['xgb_feature_names']
        self.feature_types = config['xgb_feature_types']
        self.nthread = config['xgb_nthread']

        # train params
        self.params = config['xgb_params']
        self.num_boost_round = config['xgb_num_boost_round']
        self.evals = ()
        self.obj = config['xgb_obj']
        self.feval = config['xgb_feval']
        self.maximize = config['xgb_maximize']
        self.early_stopping_rounds = config['xgb_early_stopping_rounds']
        self.evals_result = {}
        self.verbose_eval = config['xgb_verbose_eval']
        self.callbacks = None

        # evaluator
        self.eval_type = config['eval_type']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.valid_metric = config['valid_metric'].lower()

        self.evaluator = ProxyEvaluator(config)

        # model saved
        self.checkpoint_dir = config['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        saved_model_file = '{}-{}.pth'.format(self.config['model'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

    def _interaction_to_DMatrix(self, dataloader):
        r"""Convert data format from interaction to DMatrix

        Args:
            dataloader (XgboostDataLoader): xgboost dataloader.
        Returns:
            DMatrix: Data in the form of 'DMatrix'.
        """
        interaction = dataloader.dataset[:]
        interaction_np = interaction.numpy()
        cur_data = np.array([])
        columns = []
        for key, value in interaction_np.items():
            value = np.resize(value, (value.shape[0], 1))
            if key != self.label_field:
                columns.append(key)
                if cur_data.shape[0] == 0:
                    cur_data = value
                else:
                    cur_data = np.hstack((cur_data, value))

        if self.convert_token_to_onehot == True:
            from scipy import sparse
            from scipy.sparse import dok_matrix
            convert_col_list = dataloader.dataset.convert_col_list
            hash_count = dataloader.dataset.hash_count

            new_col = cur_data.shape[1] - len(convert_col_list)
            for key, values in hash_count.items():
                new_col = new_col + values
            onehot_data = dok_matrix((cur_data.shape[0], new_col))

            cur_j = 0
            new_j = 0

            for key in columns:
                if key in convert_col_list:
                    for i in range(cur_data.shape[0]):
                        onehot_data[i, int(new_j + cur_data[i, cur_j])] = 1
                    new_j = new_j + hash_count[key] - 1
                else:
                    for i in range(cur_data.shape[0]):
                        onehot_data[i, new_j] = cur_data[i, cur_j]
                cur_j = cur_j + 1
                new_j = new_j + 1

            cur_data = sparse.csc_matrix(onehot_data)

        return self.xgb.DMatrix(
            data=cur_data,
            label=interaction_np[self.label_field],
            weight=self.weight,
            base_margin=self.base_margin,
            missing=self.missing,
            silent=self.silent,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            nthread=self.nthread
        )

    def _train_at_once(self, train_data, valid_data):
        r"""

        Args:
            train_data (XgboostDataLoader): XgboostDataLoader, which is the same with GeneralDataLoader.
            valid_data (XgboostDataLoader): XgboostDataLoader, which is the same with GeneralDataLoader.
        """
        self.dtrain = self._interaction_to_DMatrix(train_data)
        self.dvalid = self._interaction_to_DMatrix(valid_data)
        self.evals = [(self.dtrain, 'train'), (self.dvalid, 'valid')]
        self.model = self.xgb.train(
            self.params, self.dtrain, self.num_boost_round, self.evals, self.obj, self.feval, self.maximize,
            self.early_stopping_rounds, self.evals_result, self.verbose_eval, self.xgb_model, self.callbacks
        )

        self.model.save_model(self.saved_model_file)
        self.xgb_model = self.saved_model_file

    def _valid_epoch(self, valid_data):
        r"""

        Args:
            valid_data (XgboostDataLoader): XgboostDataLoader, which is the same with GeneralDataLoader.
        """
        valid_result = self.evaluate(valid_data)
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        return valid_result, valid_score

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False):
        # load model
        if self.xgb_model is not None:
            self.model.load_model(self.xgb_model)

        self.best_valid_score = 0.
        self.best_valid_result = 0.

        for epoch_idx in range(self.epochs):
            self._train_at_once(train_data, valid_data)

            if (epoch_idx + 1) % self.eval_step == 0:
                # evaluate
                valid_start_time = time()
                valid_result, valid_score = self._valid_epoch(valid_data)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)

                self.best_valid_score = valid_score
                self.best_valid_result = valid_result

        return self.best_valid_score, self.best_valid_result

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        self.eval_pred = torch.Tensor()
        self.eval_true = torch.Tensor()

        self.deval = self._interaction_to_DMatrix(eval_data)
        self.eval_true = torch.Tensor(self.deval.get_label())
        self.eval_pred = torch.Tensor(self.model.predict(self.deval))

        batch_matrix_list = [[torch.stack((self.eval_true, self.eval_pred), 1)]]
        result = self.evaluator.evaluate(batch_matrix_list, eval_data)
        return result
