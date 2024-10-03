import logging
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn

import conditional_sampling
import sgm_utils
from .bdc_module import *


class MetaTemplate(nn.Module):
    def __init__(self, params, model_func, n_way, n_support, change_way=True):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = params.n_query  
        self.n_add = params.n_add
        self.feature = model_func
        self.change_way = change_way  # some methods allow different_way classification during training and test
        self.params = params

    @abstractmethod
    def set_forward(self, x, is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    @abstractmethod
    def feature_forward(self, x, sigmas):
        pass


    def forward(self, x):
        out = self.feature.forward(x)
        return out

    def parse_feature(self, x, sigmas, is_feature):
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_add + self.n_query), *x.size()[2:])
            x = self.feature.forward(x, sigmas)
            z_all = self.feature_forward(x) 
            z_all = z_all.view(self.n_way, self.n_support + self.n_add + self.n_query, -1)
        
        z_support = z_all[:, :self.n_support]
        z_add = z_all[:,self.n_support:self.n_support+self.n_add]
        z_support_add = z_all[:,:self.n_support+self.n_add]
        z_query = z_all[:, self.n_support+self.n_add:]

        return z_support, z_add, z_support_add, z_query
    
    def correct(self, x):
        scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)
    
    
    def train_loop_gen(self, params, train_loader, cls_optimizer, state, sgm_dict, config, rank):
        avg_loss = 0
        acc_all = []
        auroc_all = []
        iter_num = len(train_loader)
        sde = sgm_dict['sde']
        sampling_eps = sgm_dict['sampling_eps']
        inverse_scaler = sgm_dict['inverse_scaler']
        score_model = state['model']
        score_optimizer = state['optimizer']
        n_classes = params.train_n_way
        sampling_shape = (n_classes * self.n_add, config.data.num_channels, config.data.image_size, config.data.image_size)

        device = f'cuda:{rank}'

        sampling_fn = conditional_sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, device)

        for i, (x, l) in enumerate(train_loader):  # x size: [4, 11, 1, 128, 128] # EACH BATCH IS AN EPISODE
            logging.info(f"Episode {str(i)}")
            x = x.to(device)
            l = l.to(device)
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)

            cls_optimizer.zero_grad(set_to_none=True)
            score_optimizer.zero_grad(set_to_none=True)

            x_support = x[:, :params.n_shot, :, :, :]
            l_support = l[:, :params.n_shot]
            y = l[:, 0].cpu().numpy()
            y = torch.from_numpy(np.repeat(y, params.n_add, axis=0)).to(device)

            x_support = x_support.contiguous().view(self.n_way * self.n_support, *x_support.size()[2:]).to(device)
            l_support = l_support.contiguous().view(self.n_way * self.n_support).to(device)

            # Score model loss
            score_loss = sgm_utils.score_loss_fn(sde, score_model, x_support, l_support, sampling_eps, train=True)
            score_loss.backward()
            score_optimizer.step()
            state['ema'].update(score_model.parameters())

            # Generation phase: sampling from score model

            # Initialize random x_add for all classes
            x_add_all_classes = torch.zeros([n_classes, params.n_add, *x.size()[2:]], device=device)
            x_class = torch.cat((x[:, :params.n_shot, :, :, :], x_add_all_classes, x[:, params.n_shot:, :, :, :]), dim=1)  # initial x to give to the classifier (support + query + generated) [(4,(1+1+10),128,128]
            ema = state['ema']
            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())
            x_add, _ = sampling_fn(score_model, y) # generate new instances for each class
            x_add = x_add.contiguous().view(self.n_way, self.n_add, x_add.shape[1], x_add.shape[2], x_add.shape[3])
            ema.restore(score_model.parameters())
            x_class[:, self.n_support:self.n_support + self.n_add, :, :, :] = x_add  # substitute new instance for that label in position params.n_shot (i.e. after the support instances)
           
            # Compute classifier loss and output
            correct_this, count_this, cls_loss, _, auroc = self.set_forward_loss(x_class)

            acc_all.append(correct_this / count_this * 100)
            auroc_all.append(auroc)

            cls_loss.backward()
            cls_optimizer.step()
            total_loss = score_loss.item() + cls_loss.item() 
            avg_loss = avg_loss + total_loss

        acc_all = np.asarray(acc_all)
        auroc_all = np.asarray(auroc_all)
        acc_mean = np.mean(acc_all)
        auroc_mean = np.mean(auroc_all)

        return avg_loss / iter_num, cls_loss / iter_num, score_loss / iter_num, acc_mean, auroc_mean

    def test_loop_gen(self, params, eval_loader, state, sgm_dict, config, rank):
        avg_loss = 0
        acc_all = []
        auroc_all = []
        iter_num = len(eval_loader)
        sde = sgm_dict['sde']
        sampling_eps = sgm_dict['sampling_eps']
        inverse_scaler = sgm_dict['inverse_scaler']
        score_model = state['model']
        n_classes = params.train_n_way
        sampling_shape = (n_classes*self.n_add,config.data.num_channels,config.data.image_size,config.data.image_size)

        device = f'cuda:{rank}'

        sampling_fn = conditional_sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, device)

        with torch.no_grad():
            for i, (x, l) in enumerate(eval_loader): # x size: [4, 11, 1, 128, 128] # EACH BATCH IS AN EPISODE
                x = x.to(device)
                l = l.to(device)
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x.size(0)
                    
                x_support = x[:,:params.n_shot,:,:,:]
                l_support = l[:,:params.n_shot]
                y = l[:,0].cpu().numpy()
                y = torch.from_numpy(np.repeat(y,params.n_add,axis=0)).to(device)
                
                x_support = x_support.contiguous().view(self.n_way * self.n_support, *x_support.size()[2:]).to(device)
                l_support = l_support.contiguous().view(self.n_way * self.n_support).to(device)
                
                ## Score model loss ##
                eval_score_loss = sgm_utils.score_loss_fn(sde, score_model, x_support, l_support, sampling_eps, train=True)

                ## Generation phase: sampling from score model ##

                # Initialize random x_add for all classes
                x_add_all_classes = torch.zeros([n_classes,params.n_add,*x.size()[2:]], device=device)
                x_class = torch.cat((x[:,:params.n_shot,:,:,:],x_add_all_classes,x[:,params.n_shot:,:,:,:]),dim=1)  # initial x to give to the classifier (support + query + generated) [(4,(1+1+10),128,128]
                ema = state['ema']
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                x_add,_ = sampling_fn(score_model,y)
                x_add = x_add.contiguous().view(self.n_way,self.n_add,x_add.shape[1],x_add.shape[2],x_add.shape[3])
                ema.restore(score_model.parameters())
                x_class[:,self.n_support:self.n_support+self.n_add,:,:,:]=x_add # substitute new instance for that label in position params.n_shot (i.e. after the support instances)

                # Compute classifier loss and output
                correct_this, count_this, eval_cls_loss, _, auroc = self.set_forward_loss(x_class)

            acc_all.append(correct_this / count_this * 100)
            auroc_all.append(auroc)

            total_loss = eval_score_loss.item() + eval_cls_loss.item() #TODO capire come gestire le due loss
            avg_loss += total_loss

        acc_all = np.asarray(acc_all)
        auroc_all = np.asarray(auroc_all)
        acc_mean = np.mean(acc_all)
        auroc_mean  = np.mean(auroc_all)

        return avg_loss / iter_num, eval_cls_loss /iter_num, eval_score_loss /iter_num, acc_mean, auroc_mean, x_add

    
   