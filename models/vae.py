from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.activations import get_activation
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
    

class KumoSafeConv3d(nn.Conv3d):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        memory_count = torch.prod(torch.tensor(input.shape)).item() * 2 / 1024**3

        if memory_count > 2:
            kernel_size = self.kernel_size[0]
            part_num = int(memory_count / 2) + 1
            input_chunks = torch.chunk(input, part_num, dim=2)

            if kernel_size > 1:
                input_chunks = [input_chunks[0]] + [torch.cat((input_chunks[i - 1][:, :, -kernel_size + 1 :], input_chunks[i]), dim=2)for i in range(1, len(input_chunks))]
            output_chunks = []
            for input_chunk in input_chunks:
                output_chunks.append(super().forward(input_chunk))
                
            output = torch.cat(output_chunks, dim=2)
            return output
        else:
            return super().forward(input)


class KumoCausalConv3d(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "constant",
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.height_pad = height_pad
        self.width_pad = width_pad
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        self.temporal_dim = 2
        self.time_kernel_size = time_kernel_size

        stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = KumoSafeConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.conv_cache = None


    def fake_context_parallel_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        kernel_size = self.time_kernel_size
        if kernel_size > 1:
            cached_inputs = ([self.conv_cache] if self.conv_cache is not None else [inputs[:, :, :1]] * (kernel_size - 1))
            inputs = torch.cat(cached_inputs + [inputs], dim=2)
            
        return inputs


    def _clear_fake_context_parallel_cache(self):
        del self.conv_cache
        self.conv_cache = None


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.fake_context_parallel_forward(inputs)
        self._clear_fake_context_parallel_cache()
        self.conv_cache = inputs[:, :, -self.time_kernel_size + 1 :].clone()
        padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
        inputs = F.pad(inputs, padding_2d, mode="constant", value=0)
        output = self.conv(inputs)
        
        return output
    

class KumoSpatialNorm3D(nn.Module):

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        groups: int = 32,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=groups, eps=1e-6, affine=True)
        self.conv_y = KumoCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1)
        self.conv_b = KumoCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1)


    def forward(self, f: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        if f.shape[2] > 1 and f.shape[2] % 2 == 1:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            z_first, z_rest = zq[:, :, :1], zq[:, :, 1:]
            z_first = F.interpolate(z_first, size=f_first_size)
            z_rest = F.interpolate(z_rest, size=f_rest_size)
            zq = torch.cat([z_first, z_rest], dim=2)
        else:
            zq = F.interpolate(zq, size=f.shape[-3:])

        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        
        return new_f
    

class KumoResnetBlock3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "first",
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = get_activation(non_linearity)
        
        if spatial_norm_dim is None:
            self.norm1 = nn.GroupNorm(num_channels=in_channels, num_groups=groups, eps=eps)
            self.norm2 = nn.GroupNorm(num_channels=out_channels, num_groups=groups, eps=eps)
        else:
            self.norm1 = KumoSpatialNorm3D(f_channels=in_channels, zq_channels=spatial_norm_dim, groups=groups)
            self.norm2 = KumoSpatialNorm3D(f_channels=out_channels, zq_channels=spatial_norm_dim, groups=groups)

        self.conv1 = KumoCausalConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = KumoCausalConv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode)

        if self.in_channels != self.out_channels:
            self.conv_shortcut = KumoSafeConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)


    def forward(
        self,
        inputs: torch.Tensor,
        zq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = inputs

        hidden_states = self.norm1(hidden_states, zq)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states, zq)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        
        if self.in_channels != self.out_channels:
            inputs = self.conv_shortcut(inputs)
        hidden_states = hidden_states + inputs
        
        return hidden_states


class KumoDownsample3D(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 0,
        compress_time: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.compress_time = compress_time


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.compress_time:
            batch_size, channels, frames, height, width = x.shape
            x = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, channels, frames)

            if x.shape[-1] % 2 == 1:
                x_first, x_rest = x[..., 0], x[..., 1:]
                if x_rest.shape[-1] > 0:
                    x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)
                x = torch.cat([x_first[..., None], x_rest], dim=-1)
                x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)
            else:
                x = F.avg_pool1d(x, kernel_size=2, stride=2)
                x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)

        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        batch_size, channels, frames, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        x = self.conv(x)
        x = x.reshape(batch_size, frames, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
        
        return x
    

class KumoUpsample3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        compress_time: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.compress_time = compress_time


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.compress_time:
            if inputs.shape[2] > 1 and inputs.shape[2] % 2 == 1:
                x_first, x_rest = inputs[:, :, 0], inputs[:, :, 1:]
                x_first = F.interpolate(x_first, scale_factor=2.0)
                x_rest = F.interpolate(x_rest, scale_factor=2.0)
                x_first = x_first[:, :, None, :, :]
                inputs = torch.cat([x_first, x_rest], dim=2)
            elif inputs.shape[2] > 1:
                inputs = F.interpolate(inputs, scale_factor=2.0)
            else:
                inputs = inputs.squeeze(2)
                inputs = F.interpolate(inputs, scale_factor=2.0)
                inputs = inputs[:, :, None, :, :]
        else:
            b, c, t, h, w = inputs.shape
            inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            inputs = F.interpolate(inputs, scale_factor=2.0)
            inputs = inputs.reshape(b, t, c, *inputs.shape[2:]).permute(0, 2, 1, 3, 4)

        b, c, t, h, w = inputs.shape
        inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        inputs = self.conv(inputs)
        inputs = inputs.reshape(b, t, *inputs.shape[1:]).permute(0, 2, 1, 3, 4)

        return inputs


class KumoDownBlock3D(nn.Module):
   
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_padding: int = 0,
        compress_time: bool = False,
        pad_mode: str = "first",
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channel = in_channels if i == 0 else out_channels
            resnets.append(
                KumoResnetBlock3D(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    dropout=dropout,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    pad_mode=pad_mode,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([KumoDownsample3D(out_channels, out_channels, padding=downsample_padding, compress_time=compress_time)])
        self.gradient_checkpointing = False


    def forward(
        self,
        hidden_states: torch.Tensor,
        zq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, zq)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class KumoMidBlock3D(nn.Module):
    
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "first",
    ):
        super().__init__()

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                KumoResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    spatial_norm_dim=spatial_norm_dim,
                    non_linearity=resnet_act_fn,
                    pad_mode=pad_mode,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        self.gradient_checkpointing = False


    def forward(
        self,
        hidden_states: torch.Tensor,
        zq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, zq)

        return hidden_states


class KumoUpBlock3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        spatial_norm_dim: int = 16,
        add_upsample: bool = True,
        upsample_padding: int = 1,
        compress_time: bool = False,
        pad_mode: str = "first",
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channel = in_channels if i == 0 else out_channels
            resnets.append(
                KumoResnetBlock3D(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    dropout=dropout,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    spatial_norm_dim=spatial_norm_dim,
                    pad_mode=pad_mode,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([KumoUpsample3D(out_channels, out_channels, padding=upsample_padding, compress_time=compress_time)])
        self.gradient_checkpointing = False


    def forward(
        self,
        hidden_states: torch.Tensor,
        zq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, zq)
            
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class KumoEncoder3D(nn.Module):
    
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        down_block_types: Tuple[str, ...] = (
            "KumoDownBlock3D",
            "KumoDownBlock3D",
            "KumoDownBlock3D",
            "KumoDownBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        dropout: float = 0.0,
        pad_mode: str = "first",
        temporal_compression_ratio: float = 4,
    ):
        super().__init__()

        temporal_compress_level = int(np.log2(temporal_compression_ratio))
        self.conv_in = KumoCausalConv3d(in_channels, block_out_channels[0], kernel_size=3, pad_mode=pad_mode)
        self.down_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        for i, _ in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level

            down_block = KumoDownBlock3D(
                in_channels=input_channel,
                out_channels=output_channel,
                dropout=dropout,
                num_layers=layers_per_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                add_downsample=not is_final_block,
                compress_time=compress_time,
            )
            self.down_blocks.append(down_block)

        self.mid_block = KumoMidBlock3D(
            in_channels=block_out_channels[-1],
            dropout=dropout,
            num_layers=2,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            pad_mode=pad_mode,
        )

        self.norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[-1], eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = KumoCausalConv3d(block_out_channels[-1], 2 * out_channels, kernel_size=3, pad_mode=pad_mode)
        self.gradient_checkpointing = False


    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(sample)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states, None)
        hidden_states = self.mid_block(hidden_states, None)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        
        return hidden_states


class KumoDecoder3D(nn.Module):
    
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = (
            "KumoUpBlock3D",
            "KumoUpBlock3D",
            "KumoUpBlock3D",
            "KumoUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        dropout: float = 0.0,
        pad_mode: str = "first",
        temporal_compression_ratio: float = 4,
    ):
        super().__init__()

        reversed_block_out_channels = list(reversed(block_out_channels))
        self.conv_in = KumoCausalConv3d(in_channels, reversed_block_out_channels[0], kernel_size=3, pad_mode=pad_mode)
        self.mid_block = KumoMidBlock3D(
            in_channels=reversed_block_out_channels[0],
            num_layers=2,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            spatial_norm_dim=in_channels,
            pad_mode=pad_mode,
        )
        self.up_blocks = nn.ModuleList([])
        output_channel = reversed_block_out_channels[0]
        temporal_compress_level = int(np.log2(temporal_compression_ratio))

        for i, _ in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level

            up_block = KumoUpBlock3D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                dropout=dropout,
                num_layers=layers_per_block + 1,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                spatial_norm_dim=in_channels,
                add_upsample=not is_final_block,
                compress_time=compress_time,
                pad_mode=pad_mode,
            )
            prev_output_channel = output_channel
            self.up_blocks.append(up_block)

        self.norm_out = KumoSpatialNorm3D(reversed_block_out_channels[-1], in_channels, groups=norm_num_groups)
        self.conv_act = nn.SiLU()
        self.conv_out = KumoCausalConv3d(reversed_block_out_channels[-1], out_channels, kernel_size=3, pad_mode=pad_mode)
        self.gradient_checkpointing = False


    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(sample)
        hidden_states = self.mid_block(hidden_states, sample)
        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states, sample)

        hidden_states = self.norm_out(hidden_states, sample)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        
        return hidden_states


class KumoAutoencoderKL(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    
    _supports_gradient_checkpointing = True
    _no_split_modules = ["KumoResnetBlock3D"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = (
            "KumoDownBlock3D",
            "KumoDownBlock3D",
            "KumoDownBlock3D",
            "KumoDownBlock3D",
        ),
        up_block_types: Tuple[str] = (
            "KumoUpBlock3D",
            "KumoUpBlock3D",
            "KumoUpBlock3D",
            "KumoUpBlock3D",
        ),
        block_out_channels: Tuple[int] = (128, 256, 256, 512),
        latent_channels: int = 16,
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        temporal_compression_ratio: float = 4,
        sample_height: int = 480,
        sample_width: int = 720,
    ):
        super().__init__()

        self.encoder = KumoEncoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_eps=norm_eps,
            norm_num_groups=norm_num_groups,
            temporal_compression_ratio=temporal_compression_ratio,
        )
        self.decoder = KumoDecoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_eps=norm_eps,
            norm_num_groups=norm_num_groups,
            temporal_compression_ratio=temporal_compression_ratio,
        )
        self.use_slicing = False
        self.use_tiling = False
        self.num_latent_frames_batch_size = 2
        self.num_sample_frames_batch_size = 8

        self.tile_sample_min_height = sample_height // 2
        self.tile_sample_min_width = sample_width // 2
        self.tile_latent_min_height = int(self.tile_sample_min_height / (2 ** (len(self.config.block_out_channels) - 1)))
        self.tile_latent_min_width = int(self.tile_sample_min_width / (2 ** (len(self.config.block_out_channels) - 1)))
        self.tile_overlap_factor_height = 1 / 6
        self.tile_overlap_factor_width = 1 / 5


    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (KumoEncoder3D, KumoDecoder3D)):
            module.gradient_checkpointing = value


    def _clear_fake_context_parallel_cache(self):
        for _, module in self.named_modules():
            if isinstance(module, KumoCausalConv3d):
                module._clear_fake_context_parallel_cache()


    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_overlap_factor_height: Optional[float] = None,
        tile_overlap_factor_width: Optional[float] = None,
    ) -> None:
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_latent_min_height = int(self.tile_sample_min_height / (2 ** (len(self.config.block_out_channels) - 1)))
        self.tile_latent_min_width = int(self.tile_sample_min_width / (2 ** (len(self.config.block_out_channels) - 1)))
        self.tile_overlap_factor_height = tile_overlap_factor_height or self.tile_overlap_factor_height
        self.tile_overlap_factor_width = tile_overlap_factor_width or self.tile_overlap_factor_width


    def disable_tiling(self) -> None:
        self.use_tiling = False


    def enable_slicing(self) -> None:
        self.use_slicing = True


    def disable_slicing(self) -> None:
        self.use_slicing = False


    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        _, _, num_frames, height, width = x.shape
        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)
        frame_batch_size = self.num_sample_frames_batch_size
        num_batches = num_frames // frame_batch_size if num_frames > 1 else 1

        enc = []
        for i in range(num_batches):
            remaining_frames = num_frames % frame_batch_size
            start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
            end_frame = frame_batch_size * (i + 1) + remaining_frames
            x_intermediate = x[:, :, start_frame:end_frame]
            x_intermediate = self.encoder(x_intermediate)
            enc.append(x_intermediate)

        self._clear_fake_context_parallel_cache()
        enc = torch.cat(enc, dim=2)

        return enc


    @apply_forward_hook
    def encode(self, x: torch.Tensor) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)

        return AutoencoderKLOutput(latent_dist=posterior)


    def _decode(self, z: torch.Tensor) -> Union[DecoderOutput, torch.Tensor]:
        _, _, num_frames, height, width = z.shape
        if self.use_tiling and (width > self.tile_latent_min_width or height > self.tile_latent_min_height):
            return self.tiled_decode(z)

        frame_batch_size = self.num_latent_frames_batch_size
        num_batches = num_frames // frame_batch_size
        dec = []
        for i in range(num_batches):
            remaining_frames = num_frames % frame_batch_size
            start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
            end_frame = frame_batch_size * (i + 1) + remaining_frames
            z_intermediate = z[:, :, start_frame:end_frame]
            z_intermediate = self.decoder(z_intermediate)
            dec.append(z_intermediate)

        self._clear_fake_context_parallel_cache()
        dec = torch.cat(dec, dim=2)

        return DecoderOutput(sample=dec)


    @apply_forward_hook
    def decode(self, z: torch.Tensor) -> Union[DecoderOutput, torch.Tensor]:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        return DecoderOutput(sample=decoded)


    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
            
        return b


    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
            
        return b


    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        _, _, num_frames, height, width = x.shape

        overlap_height = int(self.tile_sample_min_height * (1 - self.tile_overlap_factor_height))
        overlap_width = int(self.tile_sample_min_width * (1 - self.tile_overlap_factor_width))
        blend_extent_height = int(self.tile_latent_min_height * self.tile_overlap_factor_height)
        blend_extent_width = int(self.tile_latent_min_width * self.tile_overlap_factor_width)
        row_limit_height = self.tile_latent_min_height - blend_extent_height
        row_limit_width = self.tile_latent_min_width - blend_extent_width
        frame_batch_size = self.num_sample_frames_batch_size

        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                num_batches = num_frames // frame_batch_size if num_frames > 1 else 1
                time = []
                for k in range(num_batches):
                    remaining_frames = num_frames % frame_batch_size
                    start_frame = frame_batch_size * k + (0 if k == 0 else remaining_frames)
                    end_frame = frame_batch_size * (k + 1) + remaining_frames
                    tile = x[:, :, start_frame:end_frame, i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width]
                    tile = self.encoder(tile)
                    time.append(tile)
                self._clear_fake_context_parallel_cache()
                row.append(torch.cat(time, dim=2))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent_width)
                result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
            result_rows.append(torch.cat(result_row, dim=4))
        enc = torch.cat(result_rows, dim=3)
        
        return enc


    def tiled_decode(self, z: torch.Tensor) -> Union[DecoderOutput, torch.Tensor]:
        
        _, _, num_frames, height, width = z.shape

        overlap_height = int(self.tile_latent_min_height * (1 - self.tile_overlap_factor_height))
        overlap_width = int(self.tile_latent_min_width * (1 - self.tile_overlap_factor_width))
        blend_extent_height = int(self.tile_sample_min_height * self.tile_overlap_factor_height)
        blend_extent_width = int(self.tile_sample_min_width * self.tile_overlap_factor_width)
        row_limit_height = self.tile_sample_min_height - blend_extent_height
        row_limit_width = self.tile_sample_min_width - blend_extent_width
        frame_batch_size = self.num_latent_frames_batch_size

        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                num_batches = num_frames // frame_batch_size
                time = []
                for k in range(num_batches):
                    remaining_frames = num_frames % frame_batch_size
                    start_frame = frame_batch_size * k + (0 if k == 0 else remaining_frames)
                    end_frame = frame_batch_size * (k + 1) + remaining_frames
                    tile = z[:, :, start_frame:end_frame, i : i + self.tile_latent_min_height, j : j + self.tile_latent_min_width]
                    tile = self.decoder(tile)
                    time.append(tile)
                self._clear_fake_context_parallel_cache()
                row.append(torch.cat(time, dim=2))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent_width)
                result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
            result_rows.append(torch.cat(result_row, dim=4))
        dec = torch.cat(result_rows, dim=3)

        return DecoderOutput(sample=dec)


    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z)
        
        return dec