"""
recbole.quick_start
########################
"""
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


def dyn_disentangling(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    """ A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str): model name
        dataset (str): dataset name
        config_file_list (list): config files used to modify experiment parameters
        config_dict (dict): parameters dictionary used to modify experiment parameters
        saved (bool): whether to save the model
    """
    config = Config(model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    if config['pre_model']:
        checkpoint_dir = config['checkpoint_dir']
        ensure_dir(checkpoint_dir)

        print(config['dataset'])
        if config['dataset'] == 'ml-1m':
            # unbias data
            saved_file = 'BPR-Nov-01-2021_20-40-34.pth'
            # bias data
            # saved_g_file = 'BPR-Nov-01-2021_20-55-34.pth'
        elif config['dataset'] == 'amazon-baby':
            saved_file = 'Generator-amazon-baby.pth'
        elif config['dataset'] == 'amazon-beauty':
            saved_file = 'Generator-amazon-beauty.pth'
        elif config['dataset'] == 'amazon-cell-phone':
            saved_file = 'Generator-amazon-cell-phone.pth'
        elif config['dataset'] == 'diginetica':
            saved_file = 'Generator-digineticq.pth'

        saved_file_path = os.path.join(checkpoint_dir, saved_file)
        print(saved_file_path)
        checkpoint = torch.load(saved_file_path)

    if config['pre_dis']:
        if config['dataset'] == 'ml-1m':
            #unbias
            #saved_d_file = 'Discriminator-Nov-01-2021_21-03-15.pth'
            #bias data
            saved_d_file = 'Discriminator-Nov-01-2021_21-04-54.pth'
        elif config['dataset'] == 'amazon-baby':
            saved_d_file = 'Discriminator-amazon-baby.pth'
        elif config['dataset'] == 'amazon-beauty':
            saved_d_file = 'Discriminator-amazon-beauty.pth'
        elif config['dataset'] == 'amazon-cell-phone':
            saved_d_file = 'Discriminator-amazon-cell-phone.pth'
        elif config['dataset'] == 'diginetica':
            saved_d_file = 'Discriminator-diginetica.pth'

        saved_d_file = os.path.join(checkpoint_dir, saved_d_file)
        print(saved_d_file)
        d_checkpoint = torch.load(saved_d_file)

    # dataset filtering
    dataset = create_dataset(config, gene_batch=None)
    #mp = dataset.mp
    #print("mp!!!!!!", dataset.mp)
    item_pop = dataset.item_inter_num
    n_item = dataset.item_num
    #print("!!!!!!!!", item_popularity)
    logger.info(dataset)

    item_popularity = np.full(dataset.item_num, 0)

    for key, pop in item_pop.items():
        item_popularity[key] = pop
    print("pop:", len(item_popularity))

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    d_valid_data = data_split(config, dataset)

    base_model = get_model(config['model'])(config, train_data).to(config['device'])
    logger.info(base_model)
    if config['pre_model']:
        base_model.load_state_dict(checkpoint['state_dict'])
        message_output = 'Loading model and parameters from {}'.format(saved_file)
        logger.info(message_output)

    discriminator = get_model('Discriminator')(config, dataset).to(config['device'])
    logger.info(discriminator)
    if config['pre_dis']:
        discriminator.load_state_dict(d_checkpoint['state_dict'])
        message_output = 'Loading Discriminator structure and parameters from {}'.format(saved_d_file)
        logger.info(message_output)


    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, base_model, item_popularity)
    config['model'] = 'Discriminator'
    d_trainer = get_trainer(config['MODEL_TYPE'], 'Discriminator')(config, discriminator, item_popularity)
    config['model'] = model
    TOTAL_BATCH = 50

    for total_batch in range(TOTAL_BATCH):
        print('Epoch:', total_batch)

        print("trainer's model:", trainer.model)
        #train G
        message_output = 'training base_model...'
        logger.info(message_output)

        for it in range(4):
            g_epoch_idx = total_batch * 4 + it + 1
            training_start_time = time()
            #########
            g_loss = train_dyn_epoch(trainer, discriminator, config, train_data, item_popularity, n_item)
            #g_loss = trainer._train_epoch(train_data, g_epoch_idx)

            trainer.train_loss_dict[g_epoch_idx] = sum(g_loss) if isinstance(g_loss, tuple) else g_loss
            training_end_time = time()
            g_train_loss_output = \
                trainer._generate_train_loss_output(g_epoch_idx, training_start_time, training_end_time, g_loss)
            trainer.logger.info(g_train_loss_output)

            #eval
            if (g_epoch_idx + 1) % trainer.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = trainer._valid_epoch(valid_data, show_progress=(config['show_progress']))
                test_score, test_result = trainer._valid_epoch(test_data, show_progress=(config['show_progress']))
                #valid_pop_result = trainer.evaluate_pop(valid_data)

                trainer.best_valid_score, trainer.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    trainer.best_valid_score,
                    trainer.cur_step,
                    max_step=trainer.stopping_step,
                    bigger=trainer.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (g_epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                #valid_pop_result, valid_pop_num, valid_pop_mean = trainer.evaluate_pop(valid_data)

                trainer.logger.info(valid_score_output)
                trainer.logger.info(valid_result_output)

                test_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (g_epoch_idx, valid_end_time - valid_start_time, test_score)
                test_result_output = 'valid result: \n' + dict2str(test_result)

                trainer.logger.info(test_score_output)
                trainer.logger.info(test_result_output)
                #trainer.logger.info(valid_pop_result)
                #trainer.logger.info(valid_pop_num)
                #trainer.logger.info(valid_pop_mean)
                #print("pop_valid_result:", valid_pop_result)


                if update_flag:
                    trainer._save_checkpoint(g_epoch_idx)
                    update_output = 'Saving current best: %s' % trainer.saved_model_file
                    trainer.logger.info(update_output)
                    trainer.best_valid_result = valid_result

                #if stop_flag:
                #    stop_output = 'Finished training, best eval result in epoch %d' % \
                #                  (g_epoch_idx - trainer.cur_step * trainer.eval_step)
                #    trainer.logger.info(stop_output)
                #    break
        #if stop_flag:
        #    stop_output = 'Finished training, best eval result in epoch %d' % \
        #                      (g_epoch_idx - trainer.cur_step * trainer.eval_step)
        #    trainer.logger.info(stop_output)
        #    break

        ########train D

        message_output = 'training discriminator...'
        logger.info(message_output)

        #print("d_model:", d_trainer.model)
        #print("base_model:", base_model)
        for epoch in range(1):
            epoch_idx = total_batch * 5 + epoch + 1
            training_start_time = time()
            train_loss = train_dis_epoch(d_trainer, base_model, config, train_data, item_popularity, n_item)
            d_trainer.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                d_trainer._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            d_trainer.logger.info(train_loss_output)

            #eval
            valid_start_time = time()
            #valid_score = d_trainer.dis_evaluate(d_valid_data, base_model, thresh=config['thresh1'], load_best_model=False,
            #                                show_progress=(config['show_progress']))

            valid_end_time = time()
            #valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
            #                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
            #d_trainer.logger.info(valid_score_output)

            d_trainer._save_checkpoint(epoch_idx)
            update_output = 'Saving current best: %s' % d_trainer.saved_model_file
            trainer.logger.info(update_output)

    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])
    logger.info('best valid result: {}'.format(trainer.best_valid_result))
    logger.info('test result: {}'.format(test_result))


    emb = base_model.item_embedding.weight.detach().cpu().numpy()
    thresh = config['thresh']
    pop = np.where(item_popularity >= thresh, 1, 0)
    #print("pop:", pop)
    tSNE(emb, pop, './figs/Dyn2.png')


def data_split(config, dataset):
    model_type = config['MODEL_TYPE']
    print("model_type:", model_type)
    es_str = [_.strip() for _ in config['eval_setting'].split(',')]
    es = EvalSetting(config)
    es.set_ordering_and_splitting(es_str[0])
    print("#####", es)

    built_datasets = dataset.build(es)
    train_dataset, valid_dataset, test_dataset = built_datasets

    d_value_data = GeneralDataLoader(config=config, dataset=valid_dataset, batch_size=config['eval_batch_size'])
    return d_value_data

def get_dis_loss(model, discriminator, item_popularity, n_item,  thresh):
    all_item_e = model.item_embedding.weight
    threshold = torch.tensor(np.full(n_item, thresh))
    pos_index = torch.tensor(item_popularity) >= threshold
    pop_label = torch.tensor(np.where(pos_index, 1, 0))
    #print("pop_label:", pop_label)
    loss = discriminator.calculate_loss(all_item_e, pop_label)
    return loss

def train_dis_epoch(trainer, model, config, d_train_data, item_popularity, n_item):
    discriminator = trainer.model
    dis_optimizer = optim.RMSprop(discriminator.parameters(), lr= 0.1 * config['learning_rate'])

    discriminator.train()

    iter_data = enumerate(d_train_data)

    for batch_idx, interaction in iter_data:
        interaction = interaction.to(config['device'])
        items = interaction['item_id']

        threshold = torch.tensor(np.full(len(interaction['pop']), config['thresh1']))
        pos_index = interaction['pop'].cpu() >= threshold
        pop_label = torch.tensor(np.where(pos_index, 1, 0)).to(config['device'])

        enc_1 = model.item_embedding(items)

    dis_optimizer.zero_grad()
    #losses = get_dis_loss(model, discriminator, item_popularity, n_item, config['thresh'])
    #print("loss!!!!!!!!!", losses)
    losses = discriminator.calculate_loss(enc_1, pop_label)
    total_loss = None

    if isinstance(losses, tuple):
        loss = sum(losses)
        loss_tuple = tuple(per_loss.item() for per_loss in losses)
        total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
    else:
        loss = losses
        total_loss = losses.item() if total_loss is None else total_loss + losses.item()
    trainer._check_nan(loss)

    loss.backward()

    if trainer.clip_grad_norm:
        clip_grad_norm_(trainer.model.parameters(), **trainer.clip_grad_norm)
    dis_optimizer.step()
    #print("after******************", model.item_embedding.weight)

    return total_loss


def train_dyn_epoch(trainer, discriminator, config, g_train_data, item_popularity, n_item):
    disturb_intensity = config['norm']

    model = trainer.model

    #generator_optimizer = optim.RMSprop(model.parameters(), lr=config['learning_rate'])
    #generator_optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])


    #dis_loss = get_dis_loss(model, discriminator, item_popularity, n_item, config['thresh'])
    #print("dis_loss:::::::::::", dis_loss)
    #dis_loss.backward(retain_graph=True)
    #attack_disturb = model.item_embedding.weight.grad.detach_()

    # disturb the embedding layer
    #norm = attack_disturb.norm(dim=-1, p=2)
    #norm_disturb = attack_disturb / (norm.unsqueeze(dim=-1) + 1e-10)
    #disturb = disturb_intensity * norm_disturb

    #print("disturb:", disturb)

    #model.item_embedding.weight.data = model.item_embedding.weight.data + disturb


    model.train()

    total_loss = None
    iter_data = enumerate(g_train_data)

    for batch_idx, interaction in iter_data:
        interaction = interaction.to(config['device'])

        items = interaction['item_id']
        threshold = torch.tensor(np.full(len(interaction['pop']), config['thresh1']))
        pos_index = interaction['pop'].cpu() >= threshold
        pop_label = torch.tensor(np.where(pos_index, 1, 0)).to(config['device'])
        enc_1 = model.item_embedding(items)

        loss_protected = discriminator.calculate_loss(enc_1, pop_label)
        loss_protected.backward(retain_graph=True)
        attack_disturb = model.item_embedding.weight.grad.detach_()
        norm = attack_disturb.norm(dim=-1, p=2)
        norm_disturb = attack_disturb / (norm.unsqueeze(dim=-1) + 1e-10)
        disturb = disturb_intensity * norm_disturb

        model.item_embedding.weight.data = model.item_embedding.weight.data - disturb

        trainer.optimizer.zero_grad()
        losses = model.calculate_loss(interaction)
        #losses -= 0.1 * dis_loss

        if isinstance(losses, tuple):
            loss = sum(losses)
            loss_tuple = tuple(per_loss.item() for per_loss in losses)
            total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
        else:
            loss = losses
            total_loss = losses.item() if total_loss is None else total_loss + losses.item()
        trainer._check_nan(loss)
        #print("loss:", loss)
        loss.backward()
        if trainer.clip_grad_norm:
            clip_grad_norm_(trainer.model.parameters(), **trainer.clip_grad_norm)
        trainer.optimizer.step()

    return total_loss

def tSNE(emb, pop, save_path):
    tsne = manifold.TSNE(n_components=2, init='pca', perplexity=2, n_iter=1200, early_exaggeration=200, random_state=11)
    red_features = tsne.fit_transform(emb)
    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots()

    ax.scatter(red_features[pop == 0, 0], red_features[pop == 0, 1],
               label='unpop', alpha=0.5, s=20, edgecolors='none', color="green")
    ax.scatter(red_features[pop == 1, 0], red_features[pop == 1, 1],
               label='pop', alpha=0.5, s=20, edgecolors='none', color="red")

    ax.legend()
    ax.grid(True)
    plt.savefig(save_path, format="png")

