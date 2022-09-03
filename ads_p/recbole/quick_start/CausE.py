import os, logging
from logging import getLogger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import math
from tqdm import tqdm

from typing import List
import matplotlib.pyplot as plt
from numpy.lib.function_base import select
from sklearn.decomposition import PCA
from sklearn import manifold
from matplotlib.backends.backend_pdf import PdfPages
import copy
import numpy as np
from time import time
from recbole.config import Config, EvalSetting
from recbole.data import create_dataset, data_preparation
from recbole.data.dataset import Dataset, DisDataset, GanDataset, SequentialDataset
from recbole.utils import init_logger, get_model, get_trainer, init_seed
from recbole.data.dataloader import GanDataLoader, DisDataLoader, GeneralNegSampleDataLoader, GeneralDataLoader
from recbole.sampler import KGSampler, Sampler, RepeatableSampler
from recbole.utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str, \
    DataLoaderType, KGDataLoaderState
from torch.nn.utils.clip_grad import clip_grad_norm_
from recbole.model.loss import BPRLoss, IPWLoss


def CausE(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):

    config = Config(model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config, gene_batch=None)
    train_data, unif_data, valid_data, test_data = data_preparation(config, dataset)

    n_item = dataset.item_num

    base_model = get_model(config['model'])(config, train_data).to(config['device'])
    logger.info(base_model)

    #imputation_model = get_model(config['model'])(config, dataset).to(config['device'])
    #logger.info(imputation_model)

    base_optimizer = torch.optim.Adam(base_model.parameters(), lr=config['learning_rate'])
    #imputation_optimizer = torch.optim.SGD(imputation_model.parameters(), lr=config['learning_rate'])

    item_pop = dataset.item_inter_num
    item_popularity = np.full(dataset.item_num, 0)
    for key, pop in item_pop.items():
        item_popularity[key] = pop

    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, base_model, item_popularity)

    criterion = nn.MSELoss(reduction='sum')

    for epoch_idx in range(config['epochs']):
        print('Epoch:', epoch_idx)
        iter_data = enumerate(train_data)
        iter_data_u = enumerate(unif_data)
        total_loss = None

        training_start_time = time()
        for batch_idx, interaction in iter_data:
            for batch_idx_u, interaction_u in iter_data_u:
                interaction = interaction.to(config['device'])
                #print(interaction)
                #print(interaction_u)
                interaction_u = interaction_u.to(config['device'])

                users = interaction['user_id']
                items = interaction['item_id']
                #y = interaction['rate'].to(torch.float32)
                y = interaction['label']

                users_unif = interaction_u['user_id']
                items_unif = interaction_u['item_id']
                #y_unif = interaction_u['rate'].to(torch.float32)
                y_unif = interaction_u['label']


                # train the base_model
                base_model.train()
                #items_unif = items_unif + n_item

                #users_combine = torch.cat((users, users_unif))
                #items_combine = torch.cat((items, items_unif))
                #y_combine = torch.cat((y, y_unif))

                y_hat = base_model(users, items)
                loss1 = criterion(y_hat, y)
                y_unif_hat = base_model.forward1(users_unif, items_unif)
                loss1 += 0.01 * criterion(y_unif_hat, y_unif)
                #print("loss1:", loss1)

                student_items_embedding = base_model.item_embedding.weight[items]
                teacher_items_embedding = torch.detach(base_model.item_embedding1.weight[items])
                reg = torch.sum(torch.abs(student_items_embedding - teacher_items_embedding))
                #print("reg:", reg)

                loss_base = loss1 + config['alpha'] * (base_model.l2_norm(users, items) + base_model.l2_norm1(users_unif, items_unif)) \
                    + config['beta'] * reg
                #loss_base = loss1 + model_args['weight_decay'] * (base_model.l2_norm(users,items) + base_model.l2_norm(users_unif,items_unif) \
                #   + teacher_args['weight_decay'] * reg

                base_optimizer.zero_grad()
                loss_base.backward()
                base_optimizer.step()


                if isinstance(loss_base, tuple):
                    loss = sum(loss_base)
                    loss_tuple = tuple(per_loss.item() for per_loss in loss_base)
                    total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
                else:
                    loss = loss_base
                    total_loss = loss_base.item() if total_loss is None else total_loss + loss_base.item()
                trainer._check_nan(loss)
            #print("loss:", loss)

        trainer.train_loss_dict[epoch_idx] = sum(total_loss) if isinstance(total_loss, tuple) else total_loss
        training_end_time = time()
        g_train_loss_output = \
            trainer._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, total_loss)
        trainer.logger.info(g_train_loss_output)

        if (epoch_idx + 1) % trainer.eval_step == 0:
            valid_start_time = time()
            valid_score, valid_result = trainer._valid_epoch(valid_data, show_progress=(config['show_progress']))
            test_score, test_result = trainer._valid_epoch(test_data, show_progress=(config['show_progress']))

            trainer.best_valid_score, trainer.cur_step, stop_flag, update_flag = early_stopping(
                valid_score,
                trainer.best_valid_score,
                trainer.cur_step,
                max_step=trainer.stopping_step,
                bigger=trainer.valid_metric_bigger
            )
            valid_end_time = time()
            valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                 (epoch_idx, valid_end_time - valid_start_time, valid_score)
            valid_result_output = 'valid result: \n' + dict2str(valid_result)

            #valid_pop_result, valid_pop_num, valid_pop_mean = trainer.evaluate_pop1(valid_data)
            #valid_pos_num, valid_pos_mean = trainer.evaluate_pop(valid_data)

            trainer.logger.info(valid_score_output)
            trainer.logger.info(valid_result_output)
            #trainer.logger.info(valid_pop_result)
            #trainer.logger.info(valid_pop_num)
            #trainer.logger.info(valid_pop_mean)
            #trainer.logger.info(valid_pos_num)
            #trainer.logger.info(valid_pos_mean)

            test_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                (epoch_idx, valid_end_time - valid_start_time, test_score)
            test_result_output = 'valid result: \n' + dict2str(test_result)

            trainer.logger.info(test_score_output)
            trainer.logger.info(test_result_output)


            if update_flag:
                trainer._save_checkpoint(epoch_idx)
                update_output = 'Saving current best: %s' % trainer.saved_model_file
                trainer.logger.info(update_output)
                trainer.best_valid_result = valid_result

            if stop_flag:
                stop_output = 'Finished training, best eval result in epoch %d' % \
                              (epoch_idx - trainer.cur_step * trainer.eval_step)
                trainer.logger.info(stop_output)
                break

