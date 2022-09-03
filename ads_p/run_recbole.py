# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2020/10/3, 2020/10/1
# @Author : Yupeng Hou, Zihan Lin
# @Email  : houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn


import argparse

from recbole.quick_start import run_recbole, run_dis, seq_gan, trans_gene_data, dyn_disentangling, dyn_disentangle_genre, dyn_disentangle_position, DR
from recbole.config import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BPR', help='name of models')
    parser.add_argument('--gan_model', type=str, default='Generator', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    parser.add_argument('--gene_batch', type=int, default=None, help='gene batch')


    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None

    config = Config(args.model, dataset=args.dataset, config_file_list=config_file_list, config_dict=None)
    train_mode = config['mode']
    if train_mode == 1:
        dyn_disentangling(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
    elif train_mode == 2:
        run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list, gene_batch=args.gene_batch)
    elif train_mode == 3:
        run_dis(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
    elif train_mode == 4:
        trans_gene_data(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
    elif train_mode == 5:
        DLA(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
    elif train_mode == 6:
        DR_Genre(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
    elif train_mode == 7:
        dyn_disentangle_genre(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
    elif train_mode == 8:
        dyn_disentangle_position(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
    elif train_mode == 9:
        DR(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
    elif train_mode == 10:
        HeckE(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
    elif train_mode == 11:
        AutoDebias(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
    elif train_mode == 12:
        CausE(model=args.model, dataset=args.dataset, config_file_list=config_file_list)


