import glob
import logging
import os
import random
from datetime import datetime

import conditional_sampling
import matplotlib.pyplot as plt
import network.resnet_pytorch as resnet_pytorch
import numpy as np
import tensorflow as tf
import torch
import tqdm
from PIL import Image
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter


model_dict = dict(
    Resnet18=resnet_pytorch.Resnet18)

def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x

def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    print(best_file)
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)


def set_gpu(args):
    if args.gpu == '-1':
        gpu_list = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_list = [int(x) for x in args.gpu.split(',')]
        print('use gpu:', gpu_list)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def load_model(model, dir):
    model_dict = model.state_dict()
    file_dict = torch.load(dir)['state']
    file_dict = {k: v for k, v in file_dict.items() if k in model_dict}
    model_dict.update(file_dict)
    model.load_state_dict(model_dict)
    return model

def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def restore_checkpoint(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    logging.warning(f"Loaded checkpoint")
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)
  


def train(params, loaders, cls_dict, rank):
    """
    Train the classification model.

    Args:
        params: Parsed command line arguments.
        loaders: Dictionary containing data loaders for training and evaluation.
        cls_dict: Dictionary containing classification model, loss function, and optimizer.
        rank: Rank of the current process in distributed training.

    Returns:
        cls_model: Trained classification model.
        trlog: Dictionary containing training and validation logs.
    """
    
    # Initialize training log
    trlog = {}
    trlog['args'] = vars(params)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_auroc'] = []
    trlog['val_auroc'] = []
    trlog['max_auroc'] = 0.0
    trlog['max_auroc_epoch'] = 0

    # Extract classification model and optimizer from the dictionary
    cls_model = cls_dict['cls_model']
    cls_optimizer = cls_dict['cls_optimizer']

    # Setup Tensorboard writer
    tb_dir = params.output_path
    writer = SummaryWriter(tb_dir)
    
    # Get current date and time for logging
    dateTimeObj = datetime.now()
    date_time = dateTimeObj.strftime("%Y%m%d%H%M%S")  
    
    auroc_val = 0.0
    valObj = 100

    # Training loop
    for epoch in range(params.epoch):
        # Update regularizer at specified epochs
        if epoch in cls_dict['decay_epochs']:
            cls_optimizer.update_regularizer(decay_factor=10)  # Decrease learning rate by 10x & update regularizer
        
        # Training phase
        cls_model.train()
        trainObj, _, _ = cls_model.train_loop(epoch, loaders['train_loader'], cls_optimizer, rank)
        
        # Evaluation phase
        if epoch % 1 == 0:
            cls_model.eval()
            valObj, _, auroc_val = cls_model.test_loop(loaders['eval_loader'], rank)
            _, _, auroc_train = cls_model.test_loop(loaders['train_eval_loader'], rank)

            # Save the best model based on validation AUROC
            if auroc_val > trlog['max_auroc'] and rank == 0:
                trlog['max_auroc'] = auroc_val
                trlog['max_auroc_epoch'] = epoch
                outfile = os.path.join(params.output_path, 'cls_checkpoint.pth')
                logging.info("Saving classification model...")
                torch.save({'epoch': epoch, 'state': cls_dict['cls_model'].state_dict()}, outfile)
                
            # Log validation metrics
            trlog['val_loss'].append(valObj)
            trlog['val_auroc'].append(auroc_val)
            writer.add_scalar("Loss/valid", valObj, epoch)
            writer.add_scalar("AUROC/valid", auroc_val, epoch)
            
            # Log training metrics
            logging.info(f"[{date_time}] Epoch: {epoch}, Loss/train: {trainObj}, Loss/valid: {valObj}, AUROC/train: {auroc_train}, AUROC/val: {auroc_val}\n")
            with open(os.path.join(params.output_path, "logger.txt"), 'a') as fout:
                fout.write(f"[{date_time}] Epoch: {epoch}, Loss/train: {trainObj}, Loss/valid: {valObj}, AUROC/train: {auroc_train}, AUROC/val: {auroc_val}\n")

            trlog['train_loss'].append(trainObj)
            trlog['train_auroc'].append(auroc_train)
            writer.add_scalar("Loss/train", trainObj, epoch)
            writer.add_scalar("AUROC/train", auroc_train, epoch)
                    
        # Early stopping if no improvement in validation AUROC for 10 epochs
        if epoch - trlog['max_auroc_epoch'] > 10 and rank == 0:
            logging.info("Early stop at epoch: %i" % epoch)
            return cls_model, trlog
        
    writer.flush()
    writer.close()
    
    return cls_model, trlog

def gen_train(params, loaders, cls_dict, sgm_dict, state, config, rank):
    """
    Train the classification model with sample generation.

    Args:
        params: Parsed command line arguments.
        loaders: Dictionary containing data loaders for training and evaluation.
        cls_dict: Dictionary containing classification model, loss function, and optimizer.
        sgm_dict: Dictionary containing score model, SDE, and other related parameters.
        state: State dictionary for the score model.
        config: Configuration object for the score model.
        rank: Rank of the current process in distributed training.

    Returns:
        cls_model: Trained classification model.
        trlog: Dictionary containing training and validation logs.
    """
    
    # Initialize training log
    trlog = {}
    trlog['args'] = vars(params)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_auroc'] = []
    trlog['val_auroc'] = []
    trlog['max_auroc'] = 0.0
    trlog['max_auroc_epoch'] = 0

    # Extract classification model and optimizer from the dictionary
    cls_model = cls_dict['cls_model']
    cls_optimizer = cls_dict['cls_optimizer']

    # Setup Tensorboard writer
    tb_dir = params.output_path
    writer = SummaryWriter(tb_dir)

    # Get current date and time for logging
    dateTimeObj = datetime.now()
    date_time = dateTimeObj.strftime("%Y%m%d%H%M%S")  

    auroc_val = 0.0
    valObj = 100

    # Training loop
    for epoch in range(params.epoch):
        # Update regularizer at specified epochs
        if epoch in cls_dict['decay_epochs']:
            cls_optimizer.update_regularizer(decay_factor=10)  # Decrease learning rate by 10x & update regularizer
        
        # Training phase
        cls_model.train()
        trainObj, cls_loss, score_loss, _, auroc_train = cls_model.train_loop_gen(params, loaders['train_loader'], cls_optimizer, state, sgm_dict, config, rank)

        # Evaluation phase
        if epoch % 1 == 0:
            cls_model.eval()
            valObj, eval_cls_loss, eval_score_loss, _, auroc_val, sample, fid_metric = cls_model.test_loop_gen(params, loaders['eval_loader'], state, sgm_dict, config, rank)  # Evaluation on validation set
            logging.info(f"FID: {fid_metric}")

            # Save generated samples
            sample = sample.detach().cpu().numpy()
            this_sample_dir = os.path.join(params.output_path, "sampled_images", f"epoch_{epoch}")
            if not os.path.isdir(this_sample_dir):
                os.makedirs(this_sample_dir)
            for l in range(sample.shape[0]):
                i_sample = sample[l,:,:,:].squeeze()
                i_sample = ((i_sample-i_sample.min()) * (255.0/i_sample.max())).astype(np.uint8)
                if params.metatrain_dataset == 'breakhis':
                    i_sample = i_sample.transpose(1, 2, 0)
                    i_sample = Image.fromarray(i_sample, 'RGB')
                else:
                    i_sample = Image.fromarray(i_sample)
                i_sample.save(os.path.join(this_sample_dir, f"sample_{l}.png"))

            # Save the best model based on validation AUROC
            if auroc_val > trlog['max_auroc'] and rank == 0:
                trlog['max_auroc'] = auroc_val
                trlog['max_auroc_epoch'] = epoch
                outfile = os.path.join(params.output_path, 'cls_checkpoint.pth')
                logging.info("Saving classification model...")
                torch.save({'epoch': epoch, 'state': cls_model.state_dict()}, outfile)

                logging.info("Saving score model...")
                save_checkpoint(os.path.join(params.output_path, 'score_checkpoint.pth'), state)

            # Log validation metrics
            trlog['val_loss'].append(valObj)
            trlog['val_auroc'].append(auroc_val)
            writer.add_scalar("Total loss/valid", valObj, epoch)
            writer.add_scalar("Cls loss/valid", eval_cls_loss, epoch)
            writer.add_scalar("Score loss/valid", eval_score_loss, epoch)
            writer.add_scalar("AUROC/valid", auroc_val, epoch)
            writer.add_scalar("FID", fid_metric, epoch)

            # Log training metrics
            logging.info(f"[{date_time}] Epoch: {epoch}, Loss/train: {trainObj}, Loss/valid: {valObj}, AUROC/train: {auroc_train}, AUROC/val: {auroc_val}\n")
            with open(os.path.join(params.output_path, "logger.txt"), 'a') as fout:
                fout.write(f"[{date_time}] Epoch: {epoch}, Loss/train: {trainObj}, Loss/valid: {valObj}, AUROC/train: {auroc_train}, AUROC/val: {auroc_val}\n")

            trlog['train_loss'].append(trainObj)
            trlog['train_auroc'].append(auroc_train)
            writer.add_scalar("Total loss/train", trainObj, epoch)
            writer.add_scalar("AUROC/train", auroc_train, epoch)
            writer.add_scalar("Cls loss/train", cls_loss, epoch)
            writer.add_scalar("Score loss/train", score_loss, epoch)

        # Early stopping if no improvement in validation AUROC for 10 epochs
        if epoch - trlog['max_auroc_epoch'] > 10 and rank == 0:
            logging.info("Early stop at epoch: %i" % epoch)
            return cls_model, trlog

    writer.flush()
    writer.close()

    return cls_model, trlog


def evaluate(data_loader, model, params, rank):
    """
    Evaluate the model on the given data loader.

    Args:
        data_loader: DataLoader object for the test data.
        model: The model to evaluate.
        params: Parameters for evaluation.
        rank: Device rank for multi-GPU setup.

    Returns:
        results: A tuple containing mean and standard deviation of AUROC and accuracy.
        y: Ground truth labels.
        softmax_scores: Softmax scores from the model.
    """
    auroc_all_task = []
    acc_all_task = []
    device = f'cuda:{rank}'

    for _ in range(params.test_task_nums):
        acc_all = []
        auroc_all = []
        tqdm_gen = tqdm.tqdm(data_loader)

        for n, (x, l) in enumerate(tqdm_gen):
            x = x.to(device)
            y = np.array(l[:, 0])
            repeated_y = np.repeat(y, params.n_shot)

            with torch.no_grad():
                model.n_query = params.n_query

                if params.print_features and n == 0:  # Print only for the first episode
                    scores, z_proto, z_support, z_query = model.set_forward(x, False)
                    z_real = z_support[:, :params.n_shot]
                    z_real = z_real.contiguous().view(params.test_n_way * params.n_shot, *z_real.size()[2:])
                    z_real_matrix = z_real.detach().cpu().numpy()
                    z_proto_matrix = z_proto.detach().cpu().numpy()

                    # Apply PCA to reduce number of components
                    n_components = params.n_shot * params.test_n_way
                    pca = PCA(n_components=n_components)
                    z_real_matrix = pca.fit_transform(z_real_matrix)
                    n_components = params.test_n_way
                    pca = PCA(n_components=n_components)
                    z_proto_matrix = pca.fit_transform(z_proto_matrix)

                    # Apply t-SNE
                    z_proto_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(z_proto_matrix)
                    z_real_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(z_real_matrix)

                    colors = {2: 'red', 3: 'green', 4: 'blue', 5: 'purple'}
                    legend_patches = [Patch(color=color, label=f"ISUP {label}") for label, color in colors.items()]

                    # Create scatter plot
                    plt.figure()
                    for emb, label in zip(z_real_embedded, repeated_y):
                        plt.scatter(emb[0], emb[1], label=label, color=colors[label])

                    for emb, label in zip(z_proto_embedded, y):
                        plt.scatter(emb[0], emb[1], label=y, color=colors[label], marker="v")

                    plt.legend(handles=legend_patches)
                    plt.axis('off')
                    plt.savefig(os.path.join(params.output_path, "scatter.pdf"))

                else:
                    scores = model.set_forward(x, False)

            y = np.repeat(range(params.test_n_way), params.n_query)

            # Calculate accuracy
            _, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = float(np.sum(topk_ind[:, 0] == y))
            acc = top1_correct / len(y) * 100
            acc_all.append(acc)

            # Calculate AUROC
            softmax_scores = torch.nn.Softmax(dim=1)(scores)
            if params.metatrain_dataset == 'picai' and params.test_n_way != 2:
                auroc_softmax = roc_auc_score(y, softmax_scores.detach().cpu().numpy(), multi_class='ovr')
            else:
                auroc_softmax = roc_auc_score(y, softmax_scores.detach().cpu().numpy()[:, 1])
            auroc_all.append(auroc_softmax)

        acc_all_task.append(acc_all)
        auroc_all_task.append(auroc_all)

    results = [(np.mean(auroc_all_task), np.std(auroc_all_task)), (np.mean(acc_all_task), np.std(acc_all_task))]
    return results, y, softmax_scores

def gen_evaluate(params, model, data_loader, state, sgm_dict, config, rank):
    """
    Evaluate the model with generated samples.

    Args:
        params: Parameters for evaluation.
        model: The model to evaluate.
        data_loader: DataLoader object for the test data.
        state: State dictionary containing model and optimizer states.
        sgm_dict: Dictionary containing SGM-related objects.
        config: Configuration object.
        rank: Device rank for multi-GPU setup.

    Returns:
        results: A tuple containing mean and standard deviation of AUROC and accuracy.
        y: Ground truth labels.
        softmax_scores: Softmax scores from the model.
    """
    auroc_all_task = []
    acc_all_task = []
    device = f'cuda:{rank}'
    sde = sgm_dict['sde']
    sampling_eps = sgm_dict['sampling_eps']
    inverse_scaler = sgm_dict['inverse_scaler']
    score_model = state['model']
    n_classes = params.test_n_way
    sampling_shape = (n_classes * params.n_add, config.data.num_channels, config.data.image_size, config.data.image_size)

    # Get the sampling function
    sampling_fn = conditional_sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, device)

    for _ in range(params.test_task_nums):
        auroc_all = []
        acc_all = []
        tqdm_gen = tqdm.tqdm(data_loader)

        for n, (x, l) in enumerate(tqdm_gen):
            x = x.to(device)
            labels = np.array(l[:, 0])

            with torch.no_grad():
                x_support = x[:, :params.n_shot, :, :, :]
                l_support = l[:, :params.n_shot]
                y = l[:, 0].cpu().numpy()
                y = torch.from_numpy(np.repeat(y, params.n_add, axis=0)).to(device)

                x_support = x_support.contiguous().view(params.test_n_way * params.n_shot, *x_support.size()[2:]).to(device)
                l_support = l_support.contiguous().view(params.test_n_way * params.n_shot).to(device)

                # Initialize random x_add for all classes and sample
                x_add_all_classes = torch.zeros([n_classes, params.n_add, *x.size()[2:]], device=device)
                x_class = torch.cat((x[:, :params.n_shot, :, :, :], x_add_all_classes, x[:, params.n_shot:, :, :, :]), dim=1)

                # Apply EMA to score model and sample
                ema = state['ema']
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                x_add, _ = sampling_fn(score_model, y)
                x_add = x_add.contiguous().view(params.test_n_way, params.n_add, x_add.shape[1], x_add.shape[2], x_add.shape[3])
                ema.restore(score_model.parameters())

                # Save sampled images before and after generation
                for j in range(x_class.shape[1]):
                    for c in range(x_class.shape[0]):
                        array = x_class[c, j, 0, :, :].cpu().numpy()
                        if array.max() != 0:
                            array = (array - array.min()) / (array.max() - array.min()) * 255
                            array = array.astype(np.uint8)
                            im = Image.fromarray(array)
                            im.save(os.path.join(params.output_path, "sampled_images", "with_label", f"test_prima_{j}_label_{c}.pdf"))
                x_class[:, params.n_shot:params.n_shot + params.n_add, :, :, :] = x_add
                for j in range(x_class.shape[1]):
                    for c in range(x_class.shape[0]):
                        array = x_class[c, j, 0, :, :].cpu().numpy()
                        if array.max() != 0:
                            array = (array - array.min()) / (array.max() - array.min()) * 255
                        array = array.astype(np.uint8)
                        im = Image.fromarray(array)
                        im.save(os.path.join(params.output_path, "sampled_images", "with_label", f"test_dopo_{j}_label_{c}.pdf"))

                # Model forward pass
                model.n_query = params.n_query

                if params.print_features and n == 0:  # Print only for the first episode
                    scores, z_proto, z_support, z_query = model.set_forward(x_class, False)

                    z_real = z_support[:, :params.n_shot]
                    z_real = z_real.contiguous().view(params.test_n_way * params.n_shot, *z_real.size()[2:])
                    z_add = z_support[:, params.n_shot:]
                    z_add = z_add.contiguous().view(params.test_n_way * params.n_add, *z_add.size()[2:])

                    z_real_matrix = z_real.detach().cpu().numpy()
                    z_add_matrix = z_add.detach().cpu().numpy()
                    z_proto_matrix = z_proto.detach().cpu().numpy()

                    # Apply PCA to reduce number of components
                    n_components = params.n_shot * params.test_n_way
                    pca = PCA(n_components=n_components)
                    z_real_matrix = pca.fit_transform(z_real_matrix)
                    n_components = params.test_n_way
                    pca = PCA(n_components=n_components)
                    z_proto_matrix = pca.fit_transform(z_proto_matrix)

                    # Apply t-SNE
                    z_proto_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(z_proto_matrix)
                    z_real_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(z_real_matrix)
                    z_add_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(z_add_matrix)

                    colors = {2: 'red', 3: 'green', 4: 'blue', 5: 'purple'}
                    legend_patches = [Patch(color=color, label=f"ISUP {label}") for label, color in colors.items()]

                    # Create scatter plot
                    plt.figure()
                    for emb, label in zip(z_real_embedded, np.repeat(labels, params.n_shot)):
                        plt.scatter(emb[0], emb[1], label=label, color=colors[label])

                    for emb, label in zip(z_add_embedded, np.repeat(labels, params.n_add)):
                        plt.scatter(emb[0], emb[1], label=label, color=colors[label], marker="s")

                    for emb, label in zip(z_proto_embedded, labels):
                        plt.scatter(emb[0], emb[1], label=y, color=colors[label], marker="v")

                    plt.legend(handles=legend_patches)
                    plt.axis('off')
                    plt.savefig(os.path.join(params.output_path, "scatter.pdf"))

                else:
                    scores = model.set_forward(x_class, False)

            y = np.repeat(range(params.test_n_way), params.n_query)

            # Calculate accuracy
            _, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = float(np.sum(topk_ind[:, 0] == y))
            acc = top1_correct / len(y) * 100
            acc_all.append(acc)

            # Calculate AUROC
            softmax_scores = torch.nn.Softmax(dim=1)(scores)
            if params.metatrain_dataset == 'picai' and params.test_n_way != 2:
                auroc_softmax = roc_auc_score(y, softmax_scores.detach().cpu().numpy(), multi_class='ovr')
            else:
                auroc_softmax = roc_auc_score(y, softmax_scores.detach().cpu().numpy()[:, 1])
            auroc_all.append(auroc_softmax)

        acc_all_task.append(acc_all)
        auroc_all_task.append(auroc_all)

    results = [(np.mean(auroc_all_task), np.std(auroc_all_task)), (np.mean(acc_all_task), np.std(acc_all_task))]
    return results, y, softmax_scores