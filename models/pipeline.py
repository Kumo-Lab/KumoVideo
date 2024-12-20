import inspect
from typing import List, Optional, Tuple, Union

import torch
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from .transformer import KumoTransformer3DModel
from .vae import KumoAutoencoderKL
from .scheduler import KumoDDIMScheduler


class KumoPipelineOutput(BaseOutput):
    frames: torch.Tensor


class KumoPipeline(DiffusionPipeline):

    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: KumoAutoencoderKL,
        transformer: KumoTransformer3DModel,
        scheduler: KumoDDIMScheduler,
    ):
        super().__init__()
        self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler)
        self.vae_scale_factor_spatial = 8
        self.vae_scale_factor_temporal = 4
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)


    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds


    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        prompt = prompt or ""
        negative_prompt = negative_prompt or ""

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        if do_classifier_free_guidance:
            negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds


    def prepare_latents(self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, (num_frames - 1) // self.vae_scale_factor_temporal + 1, num_channels_latents, height // self.vae_scale_factor_spatial, width // self.vae_scale_factor_spatial)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype) if latents is None else latents.to(device)
        latents = latents * self.scheduler.init_noise_sigma
        
        return latents


    def decode_latents(self, latents: torch.Tensor):
        latents = latents.permute(0, 2, 1, 3, 4)
        latents = 1 / self.vae.config.scaling_factor * latents
        frames = self.vae.decode(latents).sample
        
        return frames


    def prepare_extra_step_kwargs(self, generator):
        extra_step_kwargs = {}

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
            
        return extra_step_kwargs


    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 41,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        num_videos_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 256,
    ) -> Union[KumoPipelineOutput, Tuple]:
        
        device = self._execution_device
        height = height or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        num_videos_per_prompt = 1
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(batch_size * num_videos_per_prompt, latent_channels, num_frames, height, width, prompt_embeds.dtype, device, generator, latents)
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for _, t in enumerate(timesteps):

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                )[0]
                noise_pred = noise_pred.float()

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)[0]
                latents = latents.to(prompt_embeds.dtype)
                progress_bar.update()

        video = self.decode_latents(latents)
        video = self.video_processor.postprocess_video(video=video, output_type="pil")
        
        return KumoPipelineOutput(frames=video)