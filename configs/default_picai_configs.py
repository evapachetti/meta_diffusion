import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 32
  training.n_iters = 1300
  training.snapshot_freq = 500
  training.log_freq = 50
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 1000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.n_jitted_steps = 5
  training.reduce_mean = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.17

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 1
  evaluate.end_ckpt = 26
  evaluate.batch_size = 32
  evaluate.enable_sampling = True
  evaluate.num_samples = 5000
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'picai'
  data.image_size = 128
  data.random_flip = False
  data.uniform_dequantization = False
  data.centered = False
  data.num_channels = 1
  data.path = ''


  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_max = 90. # max noise variance (for VESDE)
  model.sigma_min = 0.01 # min noise variance (for VESDE)
  model.num_scales = 1000
  model.beta_min = 0.1 # min noise variance (for VPSDE)
  model.beta_max = 20. # max noise variance (for VPSDE)
  model.dropout = 0.1
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 500
  optim.grad_clip = 1.

  config.seed = 42  

  return config