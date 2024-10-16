# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import torch
import numpy as np
import abc
import sde_lib
from models import utils as mutils

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_corrector(name):
  return _CORRECTORS[name]



class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t, y):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t, y):
    """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t, y):
    dt = -1. / self.rsde.N
    z = torch.randn_like(x)
    drift, diffusion = self.rsde.sde(x, t, y)
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
    return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t, y):
    f, G = self.rsde.discretize(x, t, y)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[:, None, None, None] * z
    return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

  def vesde_update_fn(self, x, t, y):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
    score = self.score_fn(x, t, y)
    x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
    std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = torch.randn_like(x)
    x = x_mean + std[:, None, None, None] * noise
    return x, x_mean

  def vpsde_update_fn(self, x, t, y):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    beta = sde.discrete_betas.to(t.device)[timestep]
    score = self.score_fn(x, t, y)
    x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
    noise = torch.randn_like(x)
    x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
    return x, x_mean

  def update_fn(self, x, t, y):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(x, t, y)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(x, t, y)


@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, probability_flow=False):
    pass

  def update_fn(self, x, t, y):
    return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t, y):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    for i in range(n_steps):
      grad = score_fn(x, t, y)
      noise = torch.randn_like(x)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    return x, x_mean



@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
  """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t, y):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    std = self.sde.marginal_prob(x, t, y)[1]

    for i in range(n_steps):
      grad = score_fn(x, t, y)
      noise = torch.randn_like(x)
      step_size = (target_snr * std) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

    return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps):
    pass

  def update_fn(self, x, t, y):
    return x, x


def get_sampling_fn(config, sde, shape, inverse_scaler, eps, device):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """
  predictor = get_predictor(config.sampling.predictor.lower())
  corrector = get_corrector(config.sampling.corrector.lower())
  sampling_fn = get_pc_conditional_sampler(sde=sde,
                                    shape=shape,
                                    predictor=predictor,
                                    corrector=corrector,
                                    inverse_scaler=inverse_scaler,
                                    snr=config.sampling.snr,
                                    n_steps=config.sampling.n_steps_each,
                                    probability_flow=config.sampling.probability_flow,
                                    continuous=config.training.continuous,
                                    denoise=config.sampling.noise_removal,
                                    eps=eps,
                                    device=device)
 
  return sampling_fn


def get_pc_conditional_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                              n_steps=1, probability_flow=False, continuous=False, 
                              denoise=True, eps=1e-5, device='cuda'):
    """Class-conditional sampling with Predictor-Corrector (PC) samplers.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        score_model: represents the architecture of the score-based model.
        classifier: represents the architecture of the noise-dependent classifier.
        classifier_params: A dictionary that contains the weights of the classifier.
        shape: A sequence of integers. The expected shape of a single sample.
        predictor: A subclass of `sampling.predictor` that represents a predictor algorithm.
        corrector: A subclass of `sampling.corrector` that represents a corrector algorithm.
        inverse_scaler: The inverse data normalizer.
        snr: A `float` number. The signal-to-noise ratio for correctors.
        n_steps: An integer. The number of corrector steps per update of the predictor.
        probability_flow: If `True`, solve the probability flow ODE for sampling with the predictor.
        continuous: `True` indicates the score-based model was trained with continuous time.
        denoise: If `True`, add one-step denoising to final samples.
        eps: A `float` number. The SDE/ODE will be integrated to `eps` to avoid numerical issues.

    Returns: A pmapped class-conditional image sampler.
    """

    # A function that gives the logits of the noise-dependent classifier
    #logit_fn = sgm_utils.get_logit_fn(classifier)
    # The gradient function of the noise-dependent classifier
    #classifier_grad_fn = sgm_utils.get_classifier_grad_fn(logit_fn)


    def conditional_predictor_update_fn(score_model, x, t, y):#, labels):
        """The predictor update function for class-conditional sampling."""
        score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous=continuous)
     
        def total_grad_fn(x, t, y):
            #ve_noise_scale = sde.marginal_prob(x, t)[1] # std of distribution at time t
            return score_fn(x, t, y) #+ classifier_grad_fn(x, ve_noise_scale,labels)[0] # [0] otherwise is a tuple of a tensor

        if predictor is None:
            predictor_obj = None  # Replace with your NonePredictor implementation
        else:
            predictor_obj = predictor(sde, total_grad_fn, probability_flow)
        x,x_mean = predictor_obj.update_fn(x, t, y)
        return x,x_mean

    def conditional_corrector_update_fn(score_model, x, t, y):#, labels):
        """The corrector update function for class-conditional sampling."""
        score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous=continuous)
     
        def total_grad_fn(x, t, y):
            #ve_noise_scale = sde.marginal_prob(x, t)[1]
            return score_fn(x, t, y) #+ classifier_grad_fn(x,ve_noise_scale,labels)[0] # [0] otherwise is a tuple of a tensor

        if corrector is None:
            corrector_obj = None  # Replace with your NoneCorrector implementation
        else:
            corrector_obj = corrector(sde, total_grad_fn, snr, n_steps)
        x, x_mean = corrector_obj.update_fn(x, t, y)        
        return x,x_mean

    def pc_conditional_sampler(score_model, y):
        """Generate class-conditional samples with Predictor-Corrector (PC) samplers.

        Args:
            rng: A PyTorch random state.
            score_state: A dataclass object that represents the training state
            of the score-based model.
            labels: A PyTorch tensor of integers that represent the target label of each sample.

        Returns:
            Class-conditional samples.
        """
        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(shape).to(device) # sample taken for general prior distribution (e.g. Gaussian) having the shape of the original data
            timesteps = torch.linspace(sde.T, eps, sde.N, device=x.device)

            def loop_body(i, val, y):
                x, x_mean = val
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = conditional_corrector_update_fn(score_model, x, vec_t, y)
                x, x_mean = conditional_predictor_update_fn(score_model, x, vec_t, y)

                return x, x_mean
                        
            # # Run the loop
            for i in range(sde.N):
                x, x_mean = loop_body(i, (x, x), y)
            
            return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

    return pc_conditional_sampler




