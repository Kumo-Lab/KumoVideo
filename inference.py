import argparse

import torch
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.utils import export_to_video

from models.scheduler import KumoDDIMScheduler
from models.transformer import KumoTransformer3DModel
from models.vae import KumoAutoencoderKL
from models.pipeline import KumoPipeline


def main(ckpt_path, output_path, pos_prompt, neg_prompt, num_inference_steps, seed):

    vae = KumoAutoencoderKL.from_pretrained(ckpt_path, subfolder="vae", torch_dtype=torch.bfloat16)
    text_encoder = T5EncoderModel.from_pretrained(ckpt_path, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    tokenizer = T5Tokenizer.from_pretrained(ckpt_path, subfolder="tokenizer")
    scheduler = KumoDDIMScheduler.from_pretrained(ckpt_path, subfolder="scheduler")
    transformer = KumoTransformer3DModel.from_pretrained(ckpt_path, subfolder="transformer", torch_dtype=torch.bfloat16)
    vae.eval()
    text_encoder.eval()
    transformer.eval()

    pipe = KumoPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        transformer=transformer
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    video = pipe(
        prompt=pos_prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).frames[0]

    export_to_video(video, output_path, fps=8)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--pos_prompt", type=str, required=True)
    parser.add_argument("--neg_prompt", type=str, default="")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    main(args.ckpt_path, args.output_path, args.pos_prompt, args.neg_prompt, args.num_inference_steps, args.seed)