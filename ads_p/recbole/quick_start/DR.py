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


def DR(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):

    config = Config(model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config, gene_batch=None)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    base_model = get_model(config['model'])(config, train_data).to(config['device'])
    logger.info(base_model)

    imputation_model = get_model(config['model'])(config, dataset).to(config['device'])
    logger.info(imputation_model)

    base_optimizer = torch.optim.Adam(base_model.parameters(), lr=config['learning_rate'])
    imputation_optimizer = torch.optim.Adam(imputation_model.parameters(), lr=config['learning_rate'])

    item_pop = dataset.item_inter_num
    item_popularity = np.full(dataset.item_num, 0)
    for key, pop in item_pop.items():
        item_popularity[key] = pop

    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, base_model, item_popularity)

    none_criterion = nn.MSELoss(reduction='none')
    sum_criterion = nn.MSELoss(reduction='mean')

    for epoch_idx in range(config['epochs']):
        print('Epoch:', epoch_idx)
        iter_data = enumerate(train_data)
        total_loss = None

        training_start_time = time()
        for batch_idx, interaction in iter_data:
            interaction = interaction.to(config['device'])
            user = interaction['user_id']
            item = interaction['item_id']

            position = interaction['position'].to(torch.float32)
            label = interaction['rate'].to(torch.float32)
            label = label
            #print("label:", label)

            pop = np.clip(position, 1, position.max() + 1)
            pop = pop / np.linalg.norm(pop, ord=np.inf)
            pop = 1 / pop
            pop = np.clip(pop, 1, np.median(pop))
            pop = pop / np.linalg.norm(pop, ord=np.inf)
            weight = pop
            #print("weight:", weight)

            # train the imputation_model
            imputation_model.train()
            user_e, item_e = base_model(user, item)
            base_score = torch.mul(user_e, item_e).sum(dim=1)
            e = none_criterion(label, base_score)

            user_i, item_i = imputation_model(user, item)
            e_hat = torch.mul(user_i, item_i).sum(dim=1)

            cost_e = none_criterion(e_hat, e)
            loss_imp = torch.sum(weight * cost_e)
            #print("loss_imp:", loss_imp)
            imputation_optimizer.zero_grad()
            loss_imp.backward()
            imputation_optimizer.step()

            # train the base_model
            base_model.train()

            #loss_all

            u = torch.tensor(list(set(list(user.numpy()))))
            i = torch.tensor([ii for ii in range(1, 501)])
            #print("u:", len(set(u)))

            all_pair = torch.cartesian_prod(u, i)
            u_all, i_all = all_pair[:, 0], all_pair[:, 1]
            #print("u_all:", len(u_all))
            #print("i_all:", i_all)
            user_all, item_all = base_model(u_all, i_all)
            base_score_all = torch.mul(user_all, item_all).sum(dim=1)
            base_score_all_detach = torch.detach(base_score_all)

            user_i_all, item_i_all = imputation_model(u_all, i_all)
            g_all = torch.mul(user_i_all, item_i_all).sum(dim=1)

            loss_all = sum_criterion(base_score_all, g_all + base_score_all_detach)  # sum(e_hat)
            #print("loss_all:", loss_all)
            #loss_obs
            user_e, item_e = base_model(user, item)
            base_score_obs = torch.mul(user_e, item_e).sum(dim=1)
            base_score_obs_detach = torch.detach(base_score_obs)

            user_i, item_i = imputation_model(user, item)
            g_obs = torch.mul(user_i, item_i).sum(dim=1)

            e_obs = none_criterion(base_score_obs, label)
            e_hat_obs = none_criterion(base_score_obs, g_obs + base_score_obs_detach)

            cost_obs = torch.abs(e_obs - e_hat_obs)
            loss_obs = torch.sum(weight * cost_obs)

            loss_base = loss_all + loss_obs
            #print("loss_obs:", loss_obs)

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
            # print("loss:", loss)

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


            trainer.logger.info(valid_score_output)
            trainer.logger.info(valid_result_output)

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

    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info('best valid result: {}'.format(trainer.best_valid_result))
    logger.info('test result: {}'.format(test_result))

