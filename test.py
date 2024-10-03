import argparse
import logging
import os
import warnings

import torch
import torch.optim
from matplotlib import pyplot as plt

from sklearn.metrics import auc, roc_curve

import sde_lib
from data.datamgr import SetDataManager
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


def main(rank=0, world_size=1):
    # Argument parser setup
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument('--image_size', default=128, type=int, choices=[128, 224], help='input image size, 128 for picai')
    parser.add_argument("--sde", default='subvp', type=str, choices=['subvp'], help="SDE type")
    parser.add_argument("--snr", default=0.16, type=int, help="SNR value")
    parser.add_argument('--metatrain_dataset', default='picai', choices=['picai', 'breakhis'])
    parser.add_argument('--metatest_dataset', default='picai', choices=['picai', 'breakhis'])
    parser.add_argument('--csv_path_train', default=os.path.join(os.path.dirname(os.getcwd()), "ssl_trainings/PI-CAI_dataset/csv_files/few_shot/meta_isup/meta_train.csv"), type=str, help='trainset path')
    parser.add_argument('--csv_path_val', default=os.path.join(os.path.dirname(os.getcwd()), "ssl_trainings/PI-CAI_dataset/csv_files/few_shot/meta_isup/meta_val.csv"), type=str, help='valset path')
    parser.add_argument('--csv_path_test', default=os.path.join(os.path.dirname(os.getcwd()), "ssl_trainings/PI-CAI_dataset/csv_files/few_shot/meta_isup/meta_test.csv"), type=str, help='valset path')
    parser.add_argument('--model', default='Resnet18', type=str, choices=['Resnet18'])
    parser.add_argument('--method', default='meta_deepbdc', choices=['meta_deepbdc', 'protonet', 'covnet', 'relationet'])
    parser.add_argument('--weight_method', default='normal', choices=['normal', 'v1', 'v2'], help='Whether to apply a weighting method to prototype building')
    parser.add_argument('--test_n_episode', default=6, type=int, help='number of episodes in meta train')
    parser.add_argument('--test_n_way', default=4, type=int, help='number of classes used for meta train')
    parser.add_argument('--test_task_nums', default=1, type=int, help='number of times to repeat meta test')
    parser.add_argument('--n_shot', default=1, type=int, help='number of labeled data in each class, same as n_support')
    parser.add_argument('--n_query', default=10, type=int, help='number of unlabeled data in each class')
    parser.add_argument('--n_add', default=0, type=int, help='number of support images to generate per class')
    parser.add_argument('--reduce_dim', default=256, type=int, help='the output dimension of BDC dimensionality reduction layer')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--beta_min', default=0.1, type=float, help='Beta min for SUBVPSDE')
    parser.add_argument('--beta_max', default=20, type=float, help='Beta min for SUBVPSDE')
    parser.add_argument('--num_scales', default=1000, type=int, help='Number of noise scales')
    parser.add_argument('--config', default=os.path.join(os.path.dirname(os.getcwd()), "SGM/configs/classifier_picai_configs.py"), help='Path to config file')
    parser.add_argument('--score_ckp_path', help='Path to trained score model checkpoint')
    parser.add_argument('--cls_ckp_path', help='Path to trained classifier checkpoint')
    parser.add_argument('--output_path', default=os.path.join(os.getcwd(), "output"), help='output finetuned model .tar file path')
    parser.add_argument('--generation', default=False, action="store_true", help="Whether to use generate additional samples")
    parser.add_argument('--print_features', default=False, action="store_true", help="Whether to print features into scatter plots")

    # Parse arguments
    params = parser.parse_args()

    # Set device
    device = f'cuda:{rank}'

    # Set random seed
    set_seed(params.seed)

    # Load configuration based on dataset and SDE type
    if params.metatrain_dataset == 'picai':
        from configs.subvp.picai_ddpm_continuous import get_config 
    else:
        from configs.subvp.breakhis_ddpm_continuous import get_config 
       
    config = get_config()

    # Ensure n_add is 0 if generation is not enabled
    if not params.generation:
        assert params.n_add == 0

    # Data loading
    test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
    test_datamgr = SetDataManager(params.csv_path_test, params.image_size, n_query=params.n_query, n_episode=params.test_n_episode, **test_few_shot_params)
    test_loader = test_datamgr.get_data_loader(data=params.metatrain_dataset, rank=rank, world_size=world_size)

    # Model selection
    if params.method == 'protonet':
        model = ProtoNet(params, model_dict[params.model](params), **test_few_shot_params)
    elif params.method == 'meta_deepbdc':
        model = MetaDeepBDC(params, model_dict[params.model](params), **test_few_shot_params)
    elif params.method == 'covnet':
        model = CovNet(params, model_dict[params.model](params), **test_few_shot_params)

    # Load model and set to evaluation mode
    model = model.to(device)
    model = load_model(model, params.cls_ckp_path)
    logging.info("Classifier checkpoint loaded")
    model.eval()

    # Generation setup if enabled
    if params.generation:
        sde = sde_lib.subVPSDE(beta_min=params.beta_min, beta_max=params.beta_max, N=params.num_scales)
        sampling_eps = 1e-3
        score_model = DDPM(config).to(device)
        ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
        predictor = EulerMaruyamaPredictor
        corrector = NoneCorrector
        score_optimizer = torch.optim.Adam(score_model.parameters(), lr=0)

        state = dict(optimizer=score_optimizer, model=score_model, ema=ema, step=0)
        state = restore_checkpoint(params.score_ckp_path, state, device)
        logging.info("Score model checkpoint loaded")

        inverse_scaler = get_data_inverse_scaler(config)  # data inverse normalizer
        sgm_dict = {'sde': sde, 'sampling_eps': sampling_eps, 'predictor': predictor, 'corrector': corrector, 'snr': params.snr, 'inverse_scaler': inverse_scaler}

        results_test, y, softmax_scores = gen_evaluate(params, model, test_loader, state, sgm_dict, config, rank)
    else:
        results_test, y, softmax_scores = evaluate(test_loader, model, params, rank)

    # Metrics and results logging
    metrics = ['AUROC', 'Accuracy']
    with open(os.path.join(params.output_path, 'results.txt'), 'w') as f:
        f.write("*Test*\n")
        for i, metric in enumerate(metrics):
            f.write(f"{metric}: {str(results_test[i][0])} ({str(results_test[i][1])})\n")

    # ROC curve plotting
    auroc_softmax_scores = softmax_scores.detach().cpu().numpy()
    n_classes = params.test_n_way
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y, auroc_softmax_scores[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ISUP ' + str(i + 2) + ' vs. rest (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(params.output_path, "TEST_ROC_CURVE.pdf"))
    plt.show()

if __name__ == '__main__':
    main()
