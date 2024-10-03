
import torch
import logging
import torch.nn.functional as F
import sde_lib 

def get_model_fn(model, train=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: The score model.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(x, labels, y):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.

    Returns:
      A tuple of (model output, new mutable states)
    """
        
    if not train:
      model.eval()
      return model(x, labels, y)
    else:
      model.train()
      return model(x, labels, y)

  return model_fn

def get_score_fn(sde, score_model, train=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

  Returns:
    A score function.
  """
  model_fn = get_model_fn(score_model, train=train)

  if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
    def score_fn(x, t, y):
      # Scale neural network output by standard deviation and flip sign
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999
        score = model_fn(x, labels, y)

        std = sde.marginal_prob(torch.zeros_like(x), t, y)[1]
  
        score = -score / std[:, None, None, None]#.to(score.device)
        return score

  elif isinstance(sde, sde_lib.VESDE):
    def score_fn(x, t, y):
      labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
      score = score_model(x, labels, y)
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn

def score_loss_fn(sde, score_model, batch, y, eps, train=True):
    """Compute the loss function of the score model.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    score_fn = get_score_fn(sde, score_model, train=train)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t, y)
    mean = mean.to(batch.device)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t, y)
    losses = torch.square(score * std[:, None, None, None] + z)
    losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    
    return loss

def classifier_output(sde, classifier, batch, eps):

    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    correct_this, count_this, loss, scores, auroc = classifier.set_forward_loss(batch,std)
    return correct_this, count_this, loss, scores, auroc

def get_logit_fn(classifier):
     #""" A function that gives the logits of the noise-dependent classifier. """
  def logit_fn(x, ve_noise_scale):
    return classifier.make_inference(x, ve_noise_scale) 

  return logit_fn

def get_classifier_grad_fn(logit_fn):
    #"""Create the gradient function for the classifier in use of class-conditional sampling."""
  def grad_fn(x, ve_noise_scale, labels):
    def prob_fn(x):
      logits = logit_fn(x, ve_noise_scale)
      prob = F.log_softmax(logits, dim=-1)[torch.arange(1,dtype=torch.long), labels].sum()

      return prob
    
    outputs = prob_fn(x)

    if not isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
        
    return torch.autograd.grad(outputs, x)
  return grad_fn

def conditional_corrector_update_fn(label, params, corrector, sde, score_model, classifier, x_score, x_class_reshape, t_score, t_cls, labels, snr, n_steps):
    """The corrector update function for class-conditional sampling."""
    score_fn = get_score_fn(sde, score_model, train=False)
    logit_fn = get_logit_fn(classifier)
    classifier_grad_fn = get_classifier_grad_fn(logit_fn)
    
    def total_grad_fn(x_score, x_class_reshape, t_score, t_cls):
        ve_noise_scale = sde.marginal_prob(x_class_reshape, t_cls)[1]
        score_grad = score_fn(x_score, t_score)
        cls_grad = classifier_grad_fn(x_class_reshape,ve_noise_scale,labels)[0]
        cls_grad = cls_grad.view(params.train_n_way,params.n_shot +params.n_add + params.n_query,*cls_grad.size()[1:])
        cls_grad = cls_grad[label,params.n_shot,:,:,:] # get the classifier grad only for x_add
        return score_grad + cls_grad 

    if corrector is None:
        corrector_obj = None  # Replace with your NoneCorrector implementation
    else:
        corrector_obj = corrector(sde, total_grad_fn, snr, n_steps)
    
    x,x_mean = corrector_obj.update_fn(label, params, x_score, x_class_reshape, t_score, t_cls)

    return x,x_mean

def conditional_predictor_update_fn(label, params, predictor, sde, score_model, classifier, x_score, x_class_reshape, t_score, t_cls, labels):
    score_fn = get_score_fn(sde, score_model, train=False)
    logit_fn = get_logit_fn(classifier)
    classifier_grad_fn = get_classifier_grad_fn(logit_fn)
    
    def total_grad_fn(x_score, x_class_reshape, t_score, t_cls):
        ve_noise_scale = sde.marginal_prob(x_class_reshape, t_cls)[1]
        score_grad = score_fn(x_score, t_score)
        cls_grad = classifier_grad_fn(x_class_reshape,ve_noise_scale,labels)[0]
        cls_grad = cls_grad.view(params.train_n_way,params.n_shot +params.n_add + params.n_query,*cls_grad.size()[1:])
        cls_grad = cls_grad[label,params.n_shot,:,:,:] # get the classifier grad only for x_add

        return score_grad + cls_grad # [0] otherwise is a tuple of a tensor

    if predictor is None:
            predictor_obj = None  # Replace with your NonePredictor implementation
    else:
            predictor_obj = predictor(sde, total_grad_fn, probability_flow=False)
            
    x,x_mean = predictor_obj.update_fn(label, params, x_score, x_class_reshape, t_score, t_cls)
    return x,x_mean
  
def pc_sampler(corrector, predictor, sampling_eps, x_class_reshape, x_add, label, params, sde, score_model, classifier, labels, snr,n_steps):
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    # Initial sample
    shape = (1,1,128,128)
    x = sde.prior_sampling(shape).to(x_add.device)
    timesteps = torch.linspace(sde.T, sampling_eps, sde.N, device=x_add.device)
    x_add=x
    for i in range(sde.N):
      logging.info("memory allocated: %s"%str(torch.cuda.memory_allocated()))
      t = timesteps[i]
      vec_t_cls = torch.ones(x_class_reshape.size(0), device=x_add.device) * t
      vec_t_score = torch.ones(x_add.size(0), device=x_add.device) * t
      x, _ = conditional_corrector_update_fn(label, params, corrector, sde, score_model, classifier, x_add, x_class_reshape, vec_t_score, vec_t_cls, labels, snr, n_steps)
      x, _ = conditional_predictor_update_fn(label, params, predictor, sde, score_model, classifier, x_add, x_class_reshape, vec_t_score, vec_t_cls, labels)

    return x
