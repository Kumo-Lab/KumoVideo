from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import is_torch_version
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm
from diffusers.models.embeddings import get_3d_sincos_pos_embed


class KumoLayerNormZero(nn.Module):
    
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 6 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)


    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
        
        return hidden_states, encoder_hidden_states, gate[:, None, :], enc_gate[:, None, :]


class KumoPatchEmbed(nn.Module):
    
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        embed_dim: int = 1920,
        text_embed_dim: int = 4096,
        bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 41,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 256,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.sample_frames = sample_frames
        self.temporal_compression_ratio = temporal_compression_ratio
        self.max_text_seq_length = max_text_seq_length
        self.spatial_interpolation_scale = spatial_interpolation_scale
        self.temporal_interpolation_scale = temporal_interpolation_scale

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias)
        self.text_proj = nn.Linear(text_embed_dim, embed_dim)

        pos_embedding = self._get_positional_embeddings(sample_height, sample_width, sample_frames)
        self.register_buffer("pos_embedding", pos_embedding, persistent=False)


    def _get_positional_embeddings(self, sample_height: int, sample_width: int, sample_frames: int) -> torch.Tensor:
        post_patch_height = sample_height // self.patch_size
        post_patch_width = sample_width // self.patch_size
        post_time_compression_frames = (sample_frames - 1) // self.temporal_compression_ratio + 1
        num_patches = post_patch_height * post_patch_width * post_time_compression_frames

        pos_embedding = get_3d_sincos_pos_embed(
            self.embed_dim,
            (post_patch_width, post_patch_height),
            post_time_compression_frames,
            self.spatial_interpolation_scale,
            self.temporal_interpolation_scale,
        )
        pos_embedding = torch.from_numpy(pos_embedding).flatten(0, 1)
        joint_pos_embedding = torch.zeros(1, self.max_text_seq_length + num_patches, self.embed_dim, requires_grad=False)
        joint_pos_embedding.data[:, self.max_text_seq_length :].copy_(pos_embedding)

        return joint_pos_embedding


    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
        
        text_embeds = self.text_proj(text_embeds)

        batch, num_frames, channels, height, width = image_embeds.shape
        image_embeds = image_embeds.reshape(-1, channels, height, width)
        image_embeds = self.proj(image_embeds)
        image_embeds = image_embeds.view(batch, num_frames, *image_embeds.shape[1:])
        image_embeds = image_embeds.flatten(3).transpose(2, 3)
        image_embeds = image_embeds.flatten(1, 2)
        embeds = torch.cat([text_embeds, image_embeds], dim=1).contiguous()

        pre_time_compression_frames = (num_frames - 1) * self.temporal_compression_ratio + 1
        if (self.sample_height != height or self.sample_width != width or self.sample_frames != pre_time_compression_frames):
            pos_embedding = self._get_positional_embeddings(height, width, pre_time_compression_frames)
            pos_embedding = pos_embedding.to(embeds.device, dtype=embeds.dtype)
        else:
            pos_embedding = self.pos_embedding
        embeds = embeds + pos_embedding

        return embeds
    

class KumoAttnProcessor:
    
    def __init__(self):
        pass


    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        
        batch_size, _, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states, hidden_states = hidden_states.split([text_seq_length, hidden_states.size(1) - text_seq_length], dim=1)
        
        return hidden_states, encoder_hidden_states
    

class KumoBlock(nn.Module):
    
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        self.norm1 = KumoLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm",
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=KumoAttnProcessor(),
        )
        self.attn2_1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            bias=attention_bias,
            out_bias=attention_out_bias,
            cross_attention_dim=dim,
        )
        self.attn2_2 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            bias=attention_bias,
            out_bias=attention_out_bias,
            cross_attention_dim=dim,
        )
        self.norm2 = KumoLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout, inner_dim=ff_inner_dim, bias=ff_bias)


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(hidden_states, encoder_hidden_states, temb)
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
        )
        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(hidden_states, encoder_hidden_states, temb)
        
        norm_hidden_states_attnB = self.attn2_1(norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states)
        norm_encoder_hidden_states_attnB = self.attn2_2(norm_encoder_hidden_states, encoder_hidden_states=norm_hidden_states)
        norm_hidden_states = norm_hidden_states + norm_hidden_states_attnB
        norm_encoder_hidden_states = norm_encoder_hidden_states + norm_encoder_hidden_states_attnB
        
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


class KumoTransformer3DModel(ModelMixin, ConfigMixin):
    
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 41,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 256,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.patch_embed = KumoPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=True,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        self.transformer_blocks = nn.ModuleList([
            KumoBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                activation_fn=activation_fn,
                attention_bias=attention_bias,
                norm_elementwise_affine=norm_elementwise_affine,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ])
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)
        self.gradient_checkpointing = False


    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
    ):
        batch_size, num_frames, _, height, width = hidden_states.shape
        
        timesteps = timestep
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        for _, block in enumerate(self.transformer_blocks):
            hidden_states, encoder_hidden_states = block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=emb)

        hidden_states = self.norm_final(hidden_states)
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        return (output,)