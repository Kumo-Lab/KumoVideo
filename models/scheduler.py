from typing import Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput


def rescale_zero_terminal_snr(alphas_cumprod):

    alphas_bar_sqrt = alphas_cumprod.sqrt()
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
    alphas_bar = alphas_bar_sqrt**2

    return alphas_bar


class KumoDDIMScheduler(SchedulerMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
        snr_shift_scale: float = 3.0,
    ):
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float64) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod = self.alphas_cumprod / (snr_shift_scale + (1 - snr_shift_scale) * self.alphas_cumprod)
        self.alphas_cumprod = rescale_zero_terminal_snr(self.alphas_cumprod)

        self.final_alpha_cumprod = torch.tensor(1.0)
        self.init_noise_sigma = 1.0
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))
    
    
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.config.num_train_timesteps / self.num_inference_steps
        timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
        timesteps -= 1
        self.timesteps = torch.from_numpy(timesteps).to(device)


    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        a_t = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** 0.5
        b_t = alpha_prod_t_prev**0.5 - alpha_prod_t**0.5 * a_t
        prev_sample = a_t * sample + b_t * pred_original_sample

        return DDPMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)