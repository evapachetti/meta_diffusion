import argparse
import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import sde_lib
import torch
import torch.multiprocessing as mp
import torch.optim
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from data.datamgr import SetDataManager
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from methods.covnet import CovNet
from methods.meta_deepbdc import MetaDeepBDC
from methods.protonet import ProtoNet
from sampling import EulerMaruyamaPredictor, NoneCorrector
from models.ddpm import DDPM
from models.ema import ExponentialMovingAverage
from utils import *

logging.getLogger().setLevel(logging.INFO)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "54321"  # select any idle port on your machine

    init_process_group(backend="nccl", rank=rank, world_size=world_size) # nccl required with cuda


def main(rank=0, world_size=1):
    # Argument parser setup
    parser = argparse.ArgumentParser()
    
    # Add arguments to the parser
    parser.add_argument('--image_size', default=128, type=int, choices=[128,224], help='input image size, 128 for picai')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='initial learning rate of the backbone')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay of the backbone')
    parser.add_argument("--margin", default=0.5, type=float, help="Margin")
    parser.add_argument("--epoch_decay", default=0.003, type=float, help="Epoch decay (gamma)")
    parser.add_argument('--epoch', default=2, type=int, help='Stopping epoch')
    parser.add_argument("--sde", default='subvp', type=str, choices=['subvp'], help="SDE type")
    parser.add_argument("--snr", default=0.16 , type=int, help="SNR value")
    parser.add_argument("--score_lr", default=2e-4 , type=int, help="Score model learning rate")
    parser.add_argument("--score_wd", default=0 , type=int, help="Score model weight decay")
    parser.add_argument('--metatrain_dataset', default='picai', choices=['picai','breakhis'])
    parser.add_argument('--metatest_dataset', default='picai', choices=['picai','breakhis'])
    parser.add_argument('--csv_path_train', default='', type=str, help='trainset path')
    parser.add_argument('--csv_path_val', default='', type=str, help='valset path')
    parser.add_argument('--csv_path_test', default='', type=str, help='valset path')
    parser.add_argument('--model', default='Resnet18', type=str, choices=['Resnet18'])
    parser.add_argument('--method', default='meta_deepbdc', choices=['meta_deepbdc', 'protonet','covnet'])
    parser.add_argument('--weight_method', default='normal', choices=['normal', 'v1','v2'], help='Whether to apply a weighting method to prototype building')
    parser.add_argument('--train_n_episode', default=6, type=int, help='number of episodes in meta train')
    parser.add_argument('--val_n_episode', default=3, type=int, help='number of episodes in meta val')
    parser.add_argument('--train_n_way', default=4, type=int, help='number of classes used for meta train')
    parser.add_argument('--val_n_way', default=4, type=int, help='number of classes used for meta val')
    parser.add_argument('--n_shot', default=1, type=int, help='number of labeled data in each class, same as n_support')
    parser.add_argument('--n_query', default=10, type=int, help='number of unlabeled data in each class')
    parser.add_argument('--n_add', default=0, type=int, help='number of support images to generate per class')
    parser.add_argument('--output_path', default=os.path.join(os.getcwd(),"output","0"), help='output finetuned model .tar file path')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--beta_min', default=0.1, type=float, help='Beta min for SUBVPSDE')
    parser.add_argument('--beta_max', default=20, type=float, help='Beta max for SUBVPSDE')
    parser.add_argument('--num_scales', default=1000, type=int, help='Number of noise scales')
    parser.add_argument('--reduce_dim', default=256, type=int, help='the output dimension of BDC dimensionality reduction layer')
    parser.add_argument('--ckp_path', help='Path to pre-train score model checkpoint')
    parser.add_argument('--pretrained_score_model',default=False, action="store_true",help="Whether to use a pre-trained score model")
    parser.add_argument('--generation',default=False, action="store_true",help="Whether to use generate additional samples")

    device = f'cuda:{rank}'
    # Save configuration params in a txt file
    params = parser.parse_args()
    with open(os.path.join(params.output_path,"params.txt"),'w') as par:
        d = vars(params)
        for k,v in d.items():
            par.write(str(k)+': '+str(v)+'\n')
    
    set_seed(params.seed)
    
    if params.metatrain_dataset == 'picai':
        from configs.subvp.picai_ddpm_continuous import get_config 
    else:
        from configs.subvp.breakhis_ddpm_continuous import get_config 
       
    config = get_config()
    
    if not params.generation:
        assert params.n_add == 0
  
    ### Data ###

    train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
    train_datamgr = SetDataManager(params.csv_path_train, params.image_size, n_query=params.n_query, n_episode=params.train_n_episode, **train_few_shot_params)
    train_loader = train_datamgr.get_data_loader(data=params.metatrain_dataset, rank=rank, world_size = world_size)

    test_few_shot_params = dict(n_way=params.val_n_way, n_support=params.n_shot)
    eval_datamgr = SetDataManager(params.csv_path_val, params.image_size, n_query=params.n_query, n_episode=params.val_n_episode, **test_few_shot_params)
    eval_loader = eval_datamgr.get_data_loader(data=params.metatest_dataset,rank=rank, world_size = world_size)

    train_eval_datamgr = SetDataManager(params.csv_path_train, params.image_size, n_query=params.n_query, n_episode=params.val_n_episode, **test_few_shot_params)
    train_eval_loader = train_eval_datamgr.get_data_loader(data=params.metatrain_dataset,rank=rank, world_size = world_size)

    #### Score Model and additional tools ####

    sde = sde_lib.subVPSDE(beta_min=params.beta_min, beta_max=params.beta_max, N=params.num_scales)
    sampling_eps = 1e-3
  
    ddp_setup(rank, world_size) # Setup DDP 
    
    score_model = DDPM(config).to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    predictor = EulerMaruyamaPredictor
    corrector = NoneCorrector
    score_optimizer = torch.optim.Adam(score_model.parameters(), lr=params.score_lr, weight_decay=params.score_wd)

    if params.pretrained_score_model:
        state = dict(optimizer=score_optimizer, model=score_model, ema=ema, step=0)
        state = restore_checkpoint(params.ckp_path, state,device)
        
    score_model = DDP(score_model.to(f'cuda:{rank}'),  device_ids=[rank])
    score_optimizer = torch.optim.Adam(score_model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    inverse_scaler = get_data_inverse_scaler(config) # data inverse normalizer
    sgm_dict = {'sde':sde, 'sampling_eps':sampling_eps, 'predictor':predictor, 
                'corrector':corrector, 'snr':params.snr, 'inverse_scaler':inverse_scaler}
    

    #### Classification Model, Loss function and Optimizer ####
    if params.epoch == 100:
        decay_epochs = [20,50]
    elif params.epoch == 400:
        decay_epochs = [80,200]
    else:
        decay_epochs = [2000]

    if params.method == 'protonet':
        model = ProtoNet(params, model_dict[params.model](params), **train_few_shot_params)
    elif params.method == 'meta_deepbdc':
        model = MetaDeepBDC(params, model_dict[params.model](params), **train_few_shot_params)
    elif params.method == 'covnet':
        model = CovNet(params, model_dict[params.model](params), **train_few_shot_params)
    
    cls_model = DDP(model.to(f'cuda:{rank}'), device_ids=[rank])
    cls_loss_fn = AUCMLoss()
    cls_optimizer = PESG(cls_model.parameters(), #TODO which optimizer is better to use?
            loss_fn=cls_loss_fn,
            lr=params.learning_rate, 
            momentum=0.9,
            margin=params.margin, 
            epoch_decay=params.epoch_decay, 
            weight_decay=params.weight_decay)
    
    cls_dict = {'cls_model':cls_model, 'cls_loss_fn':cls_loss_fn, 'cls_optimizer':cls_optimizer, 'decay_epochs':decay_epochs}
    loaders = {'train_loader':train_loader, 'eval_loader':eval_loader, 'train_eval_loader':train_eval_loader}
    
    #### Training ####
    with open(os.path.join(params.output_path,"logger.txt"),'w') as fout:
        fout.write("results\n")
    if params.generation:
        model, trlog = gen_train(params, loaders, cls_dict, sgm_dict, state, config, rank)
    else:
        model, trlog = train(params, loaders, cls_dict, rank)
    ###################
    
    destroy_process_group() # Cleanup DDP

    #### Plot and save results ####
    plt.figure()
    plt.plot(range(len(trlog['train_auroc'])),trlog['train_auroc'],label="Training",color='navy')
    plt.plot(range(len(trlog['val_auroc'])),trlog['val_auroc'],label="Validation",color='magenta')
    plt.legend(loc="upper left")
    plt.title('AUROC')
    plt.show()
    plt.savefig(os.path.join(params.output_path,"AUROC.pdf"))

    plt.figure()
    plt.plot(range(len(trlog['train_loss'])),trlog['train_loss'],label="Training",color='green')
    plt.plot(range(len(trlog['val_loss'])),trlog['val_loss'],label="Validation",color='red')
    plt.legend(loc="upper left")
    plt.title('Loss')
    plt.show()
    plt.savefig(os.path.join(params.output_path,"LOSS.pdf"))

    np.save(os.path.join(params.output_path,"auroc_val.npy"),np.array(trlog['val_auroc']))
    np.save(os.path.join(params.output_path,"auroc_train.npy"),np.array(trlog['train_auroc']))
    ################################
    


if __name__ == '__main__':
    world_size=torch.cuda.device_count()
    mp.spawn(main, args=[world_size], nprocs=world_size)
    main()
