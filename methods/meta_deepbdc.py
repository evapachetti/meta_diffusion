import torch
import numpy as np
from .template import MetaTemplate
from .bdc_module import BDC
from sklearn.metrics import roc_auc_score
from libauc.losses import AUCMLoss

class MetaDeepBDC(MetaTemplate):
    def __init__(self, params, model_func, n_way, n_support):
        super(MetaDeepBDC, self).__init__(params, model_func, n_way, n_support)
        self.params = params
        self.loss_fn = AUCMLoss()
        reduce_dim = params.reduce_dim
        self.feat_dim = int(reduce_dim * (reduce_dim+1) / 2)
        self.dcov = BDC(is_vec=True, input_dim=self.feature.feat_dim, dimension_reduction=reduce_dim)

    def feature_forward(self, x):
        out = self.dcov(x)
        return out

    def set_forward(self, x, sigmas, is_feature=False):
        z_support, z_add, z_support_add, z_query = self.parse_feature(x, sigmas, is_feature)
        z_proto_real = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # weighting module
        alphas = torch.ones(self.n_way,self.n_support+self.n_add, device=z_proto_real.device) 
        diff = z_add.unsqueeze(2) - z_proto_real.unsqueeze(0).unsqueeze(0)
        d = -torch.norm(diff, dim=-1)
        d = torch.permute(d,(1,0,2))
        diag_d = torch.diagonal(d, dim1=1, dim2=2)
        diag_d = torch.transpose(diag_d, 0, 1)
        alpha = 1 / diag_d
       
        alphas[:, self.n_support:self.n_support+self.n_add] = alpha
        alphas = alphas.unsqueeze(-1)
        z_support_add *= alphas # weight each support sample
        z_proto_total = z_support_add.sum(1)/alphas.sum(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        scores = self.metric(z_query, z_proto_total)
        return scores

    def set_forward_loss(self, x, sigmas=None):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = y_query.type(torch.LongTensor)
        y_query = y_query.cuda()
        y_label = np.repeat(range(self.n_way), self.n_query)
        scores = self.set_forward(x, sigmas)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)
        prob_scores = torch.nn.Softmax(dim=1)(scores)
        if self.params.metatrain_dataset == 'picai' and self.params.train_n_way != 2:
            auroc = roc_auc_score(y_label,prob_scores.detach().cpu().numpy(),multi_class='ovr')
        else:
            auroc = roc_auc_score(y_label,scores.detach().cpu().numpy()[:,1])
        scores=scores.cuda()
        return float(top1_correct), len(y_label), self.loss_fn(scores, y_query), scores, auroc
    
    def make_inference(self, x, sigmas):
        return self.set_forward(x, sigmas)

    def metric(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        
        if self.n_support > 1:
            dist = torch.pow(x - y, 2).sum(2)
            score = -dist

        else:
            score = (x * y).sum(2)

        return score
