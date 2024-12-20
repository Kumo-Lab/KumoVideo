from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from diffusers.schedulers.scheduling_lcm import LCMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor


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


class KumoLCMScheduler(SchedulerMixin, ConfigMixin):
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        snr_shift_scale: float = 3.0,
    ):
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod = self.alphas_cumprod / (snr_shift_scale + (1 - snr_shift_scale) * self.alphas_cumprod)
        self.alphas_cumprod = rescale_zero_terminal_snr(self.alphas_cumprod)

        self.final_alpha_cumprod = torch.tensor(1.0)
        self.init_noise_sigma = 1.0
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        original_inference_steps: Optional[int] = None,
        strength: int = 1.0,
    ):
        original_steps = (
            original_inference_steps if original_inference_steps is not None else self.config.original_inference_steps
        )

        k = self.config.num_train_timesteps // original_steps
        lcm_origin_timesteps = np.asarray(list(range(1, int(original_steps * strength) + 1))) * k - 1
        skipping_step = len(lcm_origin_timesteps) // num_inference_steps

        if skipping_step < 1:
            raise ValueError(
                f"The combination of `original_steps x strength`: {original_steps} x {strength} is smaller than `num_inference_steps`: {num_inference_steps}. Make sure to either reduce `num_inference_steps` to a value smaller than {int(original_steps * strength)} or increase `strength` to a value higher than {float(num_inference_steps / original_steps)}."
            )

        self.num_inference_steps = num_inference_steps

        lcm_origin_timesteps = lcm_origin_timesteps[::-1].copy()
        inference_indices = np.linspace(0, len(lcm_origin_timesteps), num=num_inference_steps, endpoint=False)
        inference_indices = np.floor(inference_indices).astype(np.int64)
        timesteps = lcm_origin_timesteps[inference_indices]

        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.long)

        self._step_index = None
        self._begin_index = None

    def get_scalings_for_boundary_condition_discrete(self, timestep):
        self.sigma_data = 0.5
        scaled_timestep = timestep / 0.1

        c_skip = self.sigma_data**2 / (scaled_timestep**2 + self.sigma_data**2)
        c_out = scaled_timestep / (scaled_timestep**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[LCMSchedulerOutput, Tuple]:
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        prev_step_index = self.step_index + 1
        if prev_step_index < len(self.timesteps):
            prev_timestep = self.timesteps[prev_step_index]
        else:
            prev_timestep = timestep

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

        predicted_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output

        denoised = c_out * predicted_original_sample + c_skip * sample

        if self.step_index != self.num_inference_steps - 1:
            noise = randn_tensor(
                model_output.shape, generator=generator, device=model_output.device, dtype=denoised.dtype
            )
            prev_sample = alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise
        else:
            prev_sample = denoised

        self._step_index += 1

        if not return_dict:
            return (prev_sample, denoised)

        return LCMSchedulerOutput(prev_sample=prev_sample, denoised=denoised)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def __len__(self):
        return self.config.num_train_timesteps

    def previous_timestep(self, timestep):
        num_inference_steps = (
            self.num_inference_steps if self.num_inference_steps else self.config.num_train_timesteps
        )
        prev_t = timestep - self.config.num_train_timesteps // num_inference_steps

        return prev_t
