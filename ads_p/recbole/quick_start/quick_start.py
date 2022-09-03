# @Time   : 2020/10/6
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
recbole.quick_start
########################
"""
import os, logging
import torch
from logging import getLogger
import numpy as np

from typing import List
import matplotlib.pyplot as plt
from numpy.lib.function_base import select
from sklearn.decomposition import PCA
from sklearn import manifold
from matplotlib.backends.backend_pdf import PdfPages

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer, init_seed, ensure_dir
from recbole.data.dataset import Dataset, DisDataset, GanDataset, SequentialDataset


def run_recbole(model=None, dataset=None, config_file_list=None, config_dict=None, gene_batch=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str): model name
        dataset (str): dataset name
        config_file_list (list): config files used to modify experiment parameters
        config_dict (dict): parameters dictionary used to modify experiment parameters
        saved (bool): whether to save the model
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    if config['add_generator'] == True and config['if_gan'] == False:
        load_generator(config, logger, gene_batch)


    # dataset filtering
    dataset = create_dataset(config, gene_batch)
    user_mp = dataset.user_mp
    item_mp = dataset.item_mp
    #print("mp!!!!!!", dataset.mp)
    item_pop = dataset.item_inter_num
    #print("!!!!!!!!", item_pop)
    logger.info(dataset)
    #if config['model'] == 'BPR_PC':
    item_popularity = np.full(dataset.item_num, 0)

    for key, pop in item_pop.items():
        item_popularity[key] = pop
    #print("item_pop:", sorted(item_popularity, reverse=True))

    sum = 0
    thresh = 0
    tot_inter = len(dataset.inter_feat['item_id'].numpy())
    #print("tot_inter:", tot_inter)
    thresh = sorted(item_popularity, reverse=True)[int(dataset.item_num / 2)]
    print("threshold!!!!!!!!!!", thresh)
    for pop in sorted(item_popularity, reverse=True):
        sum += pop
        if sum >= int(tot_inter / 2):
            thresh1 = pop
            print("threshold222222!!!!!!!!!!", thresh1)
            break

    pos, neg = 0, 0
    for i in dataset.inter_feat['item_id'].numpy():
        if item_popularity[i] >= thresh:
            pos += 1
        else:
            neg += 1
    print("pos:", pos)
    print("neg:", neg)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)


    # model loading and initialization
    model = get_model(config['model'])(config, train_data).to(config['device'])
    logger.info(model)

    # trainer loading and initialization

    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model, item_popularity)


    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    #trainer.write_rec_list('./dataset/simulation/rec_list.txt', valid_data, user_mp, item_mp)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))

    #if config['model'] == 'DICE':
    #    emb = model.item_int_embedding.weight.detach().numpy()
    #else:
    #    emb = model.item_embedding.weight.detach().numpy()
    #thresh = config['thresh']
    #pop = np.where(item_popularity >= thresh, 1, 0)
    #print("pop:", pop)
    #tSNE(emb, pop, './figs/bpr_simulation.png')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

def load_generator(config, logger, gene_batch):
    checkpoint_dir = config['checkpoint_dir']
    ensure_dir(checkpoint_dir)
    NEGATIVE_FILE = 'dataset/generator/narm_pretrain/' + 'gene.data'

    if config['dataset'] == 'ml-1m':
        saved_g_file = 'Generator-ml-1m.pth'
    elif config['dataset'] == 'amazon-baby':
        saved_g_file = 'Generator-amazon-baby.pth'
    elif config['dataset'] == 'amazon-beauty':
        saved_g_file = 'Generator-amazon-beauty.pth'
    elif config['dataset'] == 'amazon-cell-phone':
        saved_g_file = 'Generator-amazon-cell-phone.pth'
    elif config['dataset'] == 'diginetica':
        saved_g_file = 'Generator-digineticq.pth'
    elif config['dataset'] == 'gowalla':
        saved_g_file = 'Generator-Jun-10-2021_11-33-22.pth'


    #saved_g_file = 'Generator-amazon-baby.pth'
    #saved_g_file = 'Generator-Apr-13-2021_22-14-05.pth'
    #saved_g_file = 'Generator-ml-1m.pth'
    #saved_g_file = 'Generator-amazon-cell-phone.pth'
    #saved_g_file = 'Generator-digineticq.pth'

    saved_g_file = os.path.join(checkpoint_dir, saved_g_file)
    g_checkpoint = torch.load(saved_g_file)

    config['add_generator'] = False
    r_dataset = Dataset(config, gene_batch)

    hidden_size = config['hidden_size']
    emb_size = config['embedding_size']

    gene_config = config
    gene_config['embedding_size'] = 64
    gene_config['hidden_size'] = 64

    generator = get_model('Generator')(gene_config, r_dataset).to(gene_config['device'])
    logger.info(generator)

    generator.load_state_dict(g_checkpoint['state_dict'])
    message_output = 'Loading generator structure and parameters from {}'.format(saved_g_file)
    logger.info(message_output)

    #NEGATIVE_FILE = config['generator_path'] + 'gene.data'
    max_item_list_len = config['MAX_ITEM_LIST_LENGTH']
    if config['dataset'] == 'ml-1m':
        gene_num = 200000
    else:
        gene_num = 100000
    generate_samples(generator, max_item_list_len, gene_num, NEGATIVE_FILE)
    print("generate data done!!!!!", NEGATIVE_FILE)

    config['hidden_size'] = hidden_size
    config['embedding_size'] = emb_size
    config['add_generator'] = True


def generate_samples(model, seq_len, generated_num, output_file):
    samples = []
    batch_size = 100
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, seq_len, g_out=5).cpu().data.numpy().tolist()
        samples.extend(sample)
    else:
        with open(output_file, 'w') as (fout):
            for sample in samples:
                string = ' '.join([str(s) for s in sample])
                fout.write('%s\n' % string)


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict): parameters dictionary used to modify experiment parameters
        config_file_list (list): config files used to modify experiment parameters
        saved (bool): whether to save the model
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(config['model'])(config, train_data).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def tSNE(emb, pop, save_path):
    tsne = manifold.TSNE(n_components=2, init='pca', perplexity=2, n_iter=1200, early_exaggeration=200, random_state=11)
    red_features = tsne.fit_transform(emb)
    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots()

    ax.scatter(red_features[pop == 0, 0], red_features[pop == 0, 1],
               label='unpop', alpha=0.5, s=20, edgecolors='none', color="green")
    ax.scatter(red_features[pop == 1, 0], red_features[pop == 1, 1],
               label='pop', alpha=0.5, s=20, edgecolors='none', color="black")

    ax.legend()
    ax.grid(True)
    plt.savefig(save_path, format="png")