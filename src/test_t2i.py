# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained stage-2 model using DDP and
stores results for downstream metrics. For single-device sampling, use sample.py.
"""
import sys
import os
from omegaconf import OmegaConf
import yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from filelock import FileLock
import argparse
import json
import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.cuda.amp import autocast
from tqdm import tqdm
import warnings
from torchvision import transforms

from utils.model_utils import instantiate_from_config
from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_transport, Sampler
from utils.train_utils import parse_configs

from stage2 import Wrapper
from stage2.text_encoders.perceiver import PerceiverVE
from stage2.text_encoders.clip import FrozenCLIPEmbedder
from utils.data_utils import IN1KTextJsonlSampler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.fid_score import calculate_fid_given_paths
from utils.clip_score import ClipScore
import shutil
from torch.utils.data import Subset

import json

def create_npz_from_sample_folder(temp_dir, num=30_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    images = os.listdir(temp_dir)
    assert len(images) == num, f"Expected {num} samples in {temp_dir}, found {len(images)}."
    for i, image in enumerate(tqdm(images, desc="Building .npz from samples")):
        sample_pil = Image.open(os.path.join(temp_dir, image))
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    
    npz_path = f"{temp_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def main(args):
    """Run sampling with distributed execution."""
    # device
    if not torch.cuda.is_available():
        raise RuntimeError("Sampling with DDP requires at least one GPU. Use sample.py for single-device usage.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)

    # config
    exp_path = os.path.join(args.exps_dir, args.exp_name)
    config_path = os.path.join(exp_path, "config.yaml")
    ckpt_path = os.path.join(exp_path, "checkpoints", f"{args.train_step:07d}.pt")

    rae_config, model_config, textve_config, clip_config, transport_config, sampler_config, guidance_config, misc, _ = parse_configs(config_path)
    if rae_config is None or model_config is None:
        raise ValueError("Config must provide both stage_1 and stage_2 entries.")
    misc = {} if misc is None else dict(misc)

    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    use_null_text_embed = model_config.get("params", {}).get("use_null_text_embed", False)

    print(f"Using null text embed: {use_null_text_embed}")
    use_y = misc.get("use_y", False)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    if rank == 0:
        print(f"Using time_dist_shift={time_dist_shift:.4f}.")
        print(f"Expected RAE latent shape: {latent_size}")

    rae: RAE = instantiate_from_config(rae_config).to(device)
    clip: FrozenCLIPEmbedder = instantiate_from_config(clip_config).to(device)
    
    dit: Stage2ModelProtocol = instantiate_from_config(model_config)
    textve = None
    if textve_config is not None:
        if textve_config.get("target", None) is not None:
            textve: PerceiverVE = instantiate_from_config(textve_config).to(device)
    modules = {
        "dit": dit,
        "textve": textve,
        "logit_scale": torch.nn.Parameter(torch.ones([], device=device) * np.log(1/0.07)),
    }
    wrapper_model = Wrapper(**modules)
    
    clipscore_model = ClipScore(device=device)
    
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "ema" in state_dict:
        state_dict = state_dict["ema"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    wrapper_model.load_state_dict(state_dict, strict=False)
    wrapper_model.to(device)
    
    rae.eval()
    wrapper_model.eval()
    clip.eval()

    transport_params = {}
    if transport_config is not None:
        transport_params = dict(transport_config.get("params", {}))
    transport = create_transport(
        **transport_params,
        time_dist_shift=time_dist_shift,
    )
    sampler = Sampler(transport)

    sampler_config = {} if sampler_config is None else dict(sampler_config)
    sampler_mode = sampler_config.get("mode", "ODE")
    sampler_params = dict(sampler_config.get("params", {}))
    sampler_params['sampling_method'] = args.sampling_method
    sampler_params['num_steps'] = args.sampling_num_steps
    mode = sampler_mode.upper()
    if mode == "ODE":
        sample_fn = sampler.sample_ode_npe(**sampler_params)
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")

    model_fn = wrapper_model.dit.forward
    if args.cfg_method == "cfg":
        guidance_method = "cfg"
        guidance_scale = args.cfg_scale
        guidance_interval = args.cfg_interval
        if guidance_scale > 1.0:
            model_fn = wrapper_model.dit.forward_with_cfg
    elif args.cfg_method == "ag":
        guidance_method = "ag"
        guidance_scale = args.cfg_scale
        guidance_interval = args.cfg_interval
        print(f"Using autoguidance with scale {guidance_scale} and interval {guidance_interval}.")
        if args.autoguidance_exp_config is None:
            args.autoguidance_exp_config = config_path
            print("No autoguidance exp config provided, using same config as main")
        if args.autoguidance_ckpt is None:
            assert args.autoguidance_train_step is not None, "Please provide autoguidance checkpoint or train step."
            args.autoguidance_ckpt = os.path.join(exp_path, "checkpoints", f"{args.autoguidance_train_step:07d}.pt")
        if guidance_scale > 1.0:
            with open(args.autoguidance_exp_config, "r") as f:
                guid_exp_config = OmegaConf.load(f)
            guid_model_config = guid_exp_config.get("stage_2")
            guid_model: Stage2ModelProtocol = instantiate_from_config(guid_model_config).to(device)
            
            # load checkpoint
            guid_state_dict = torch.load(args.autoguidance_ckpt, map_location="cpu")
            if "ema" in guid_state_dict:
                guid_state_dict = guid_state_dict["ema"]
            elif "model" in guid_state_dict:
                guid_state_dict = guid_state_dict["model"]
            new_guid_state_dict = {}
            for k, v in guid_state_dict.items():
                if k.startswith("dit."):
                    new_guid_state_dict[k.replace("dit.", "")] = v
            guid_model.load_state_dict(new_guid_state_dict, strict=False)

            guid_model.eval()
            guid_model_forward = guid_model.forward
            model_fn = wrapper_model.dit.forward_with_autoguidance
    else:
        model_fn = wrapper_model.dit.forward
        guidance_method = "none"
        guidance_scale = 1.0
        guidance_interval = (0.0, 1.0)
        
    with torch.no_grad():        
        null_clip_latents, null_latents_and_others = clip.encode("")
        null_valid_masks = null_latents_and_others["token_mask"]
        null_clip_pooler_output = null_latents_and_others["pooler_output"]

    from collections import OrderedDict
    
    # logging info
    exp_info = OrderedDict()
    exp_info["exp_name"] = args.exp_name
    exp_info["train_step"] = args.train_step
    exp_info["num_samples"] = args.num_samples
    exp_info["guidance"] = f"{guidance_method}-{guidance_scale:.2f}"

    exp_info["sampler_mode"] = mode
    exp_info["sampling_method"] = str(sampler_params.get("sampling_method", "na"))
    exp_info["num_steps"] = int(sampler_params.get("num_steps", "na"))
    exp_info["precision"] = args.precision
    exp_info["global_seed"] = args.global_seed
    exp_info["batch_size_per_proc"] = f"bs{args.per_proc_batch_size}"

    if args.make_npz:
        folder_name = f"{args.exp_name}_{args.train_step}"
    else:
        folder_name = "-".join([f"{k}_{v}" for k, v in exp_info.items()])
    temp_dir = os.path.join(args.sample_dir, f"{folder_name}")

    if rank == 0:
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Saving temporal .png samples at {temp_dir}")
    dist.barrier()

    n = args.per_proc_batch_size
    global_batch_size = n * world_size
    
    assert args.num_samples % global_batch_size == 0
    total = 0
    
    dataset = IN1KTextJsonlSampler(
        jsonl_path=os.path.join(args.jsonl_path),
    )
    
    if args.num_samples < len(dataset):
        np.random.seed(args.global_seed)
        indices = np.random.permutation(len(dataset))
        subset_indices = indices[:args.num_samples]
        final_dataset = Subset(dataset, subset_indices)
        shuffle_sampler = True
    else:
        final_dataset = dataset
        
        
    sampler = DistributedSampler(
        final_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    
    loader = DataLoader(
        final_dataset,
        batch_size=n,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    if rank == 0:
        print(f"Starting sampling loop... Processing {args.num_samples}/{len(dataset)} total captions.")
    
    pbar = tqdm(loader, desc="Sampling Batches") if rank == 0 else loader
    
    # metrics
    local_clip_scores = []
    local_npe_scores = [] 
    for ids, captions in pbar:
        current_batch_size = min(n, len(captions))
        if os.path.exists(f"{temp_dir}/{ids[-1]}.png"):
            if rank == 0:
                print(f"Skipping already-sampled batch ending with id {ids[-1]}.")
            total += global_batch_size
            continue
        with autocast(**autocast_kwargs) and torch.no_grad():
            # 1) x0
            clip_latents, latents_and_others = clip.encode(captions)
            clip_valid_masks = latents_and_others["token_mask"]
            padding_masks = ~clip_valid_masks.to(torch.bool)
            clip_pooler_output = latents_and_others["pooler_output"]
            if textve is None: # x0 ~ N(0, I) 
                x0 = torch.randn(current_batch_size, *latent_size, device=device)
            else: # x0 ~ p_theta(.|c)                
                x0_token, _, _ = wrapper_model('textve', text_tokens=clip_latents, text_key_padding_mask=padding_masks)
                x0 = x0_token.permute(0, 2, 1).contiguous().view(current_batch_size, *latent_size)
            # 2) sample
            dit_kwargs = {"y": clip_pooler_output, "context": clip_latents, "context_mask": clip_valid_masks.to(torch.bool), "null_context": null_clip_latents, "null_context_mask": null_valid_masks.to(torch.bool)}
            # use guidnace
            if guidance_method in ["cfg", "ag"] and guidance_scale > 1.0: 
                dit_kwargs["cfg_scale"] = guidance_scale
                dit_kwargs["cfg_interval"] = guidance_interval
                if use_null_text_embed:
                    bsz = x0.shape[0]
                    null_y = null_clip_pooler_output.expand(bsz, -1)
                    null_context = null_clip_latents.expand(bsz, -1, -1)
                    null_context_mask = null_valid_masks.expand(bsz, -1)
                else:
                    null_y = torch.zeros_like(clip_pooler_output)
                    null_context = torch.zeros_like(clip_latents)
                    null_context_mask = torch.zeros_like(clip_valid_masks).to(torch.bool)
                
                x0 = torch.cat([x0, x0], dim=0)
                dit_kwargs["y"] = torch.cat([clip_pooler_output, null_y], dim=0)
                dit_kwargs["context"] = torch.cat([clip_latents, null_context], dim=0)
                dit_kwargs["context_mask"] = torch.cat([clip_valid_masks.to(torch.bool), null_context_mask], dim=0)
            
            if guidance_method == "ag" and guidance_scale > 1.0:
                dit_kwargs["additional_model_forward"] = guid_model_forward
                
            traj, path_energy = sample_fn(x0, model_fn, **dit_kwargs)
            x1_samples = traj[-1]
            
            if guidance_method in ["cfg", "ag"] and guidance_scale > 1.0:
                x1_samples = x1_samples[:current_batch_size]
                path_energy = path_energy[:current_batch_size]
                x0 = x0[:current_batch_size]

            straight_energy = (x1_samples - x0).pow(2).flatten(1).sum(dim=1)
            npe_batch = (path_energy.abs() - straight_energy).abs() / (straight_energy + 1e-8)
            local_npe_scores.extend(npe_batch.cpu().tolist())
            
            samples = rae.decode(x1_samples).clamp(0, 1)
            
            
        clip_scores = clipscore_model.calculate_clip_score(samples, captions)
        local_clip_scores.extend(clip_scores.cpu().tolist())
        samples = samples.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        for img_id, sample in zip(ids, samples):
            Image.fromarray(sample).save(f"{temp_dir}/{img_id}.png")

        total += global_batch_size
        dist.barrier()

    dist.barrier()
    
    gathered_clip_scores_list = [None] * world_size
    dist.all_gather_object(gathered_clip_scores_list, local_clip_scores)
    
    gathered_npe_scores_list = [None] * world_size
    dist.all_gather_object(gathered_npe_scores_list, local_npe_scores)
    
    if rank == 0:
        all_scores = []
        for rank_scores in gathered_clip_scores_list:
            all_scores.extend(rank_scores)
        all_scores = all_scores[:args.num_samples]
        final_clip_score = np.mean(all_scores)
        
        all_npe = []
        for rank_npe in gathered_npe_scores_list:
            all_npe.extend(rank_npe)
        all_npe = all_npe[:args.num_samples]
        final_npe_score = np.mean(all_npe)
        
        fid = calculate_fid_given_paths((args.fid_stat_npz, temp_dir),device=device)
        
        print(f"--- CLIP: {final_clip_score:.4f}, FID: {fid:.4f}, NPE: {final_npe_score:.4f} ---")
        
        metrics = {}
        metrics['clip_score'] = final_clip_score
        metrics['fid'] = fid
        metrics['npe'] = final_npe_score

        if args.make_npz:
            create_npz_from_sample_folder(temp_dir, args.num_samples)
            print("Done.")
        
        if args.remove_temp_dir:
            shutil.rmtree(temp_dir)
            print(f"Removed temporary sample directory {temp_dir}.")
        # save json
        with open(os.path.join(args.sample_dir, f"{folder_name}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
            print(f"Saved metrics to {f.name}.")
        
        # save csv
        exp_info.update(metrics)
        data = pd.DataFrame([exp_info])
        if args.csv_path is not None:
            lock = FileLock(args.csv_path +".lock")
            with lock:    
                if not os.path.exists(args.csv_path):
                    data.to_csv(args.csv_path, mode="w", header=True, index=False)
                else:
                    data.to_csv(args.csv_path, mode="a", header=False, index=False)     
                
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exps_dir", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--train_step", type=int, required=True)
    parser.add_argument("--csv_path", type=str, default=None, help="ex) test.csv")

    parser.add_argument("--sampling_method", type=str, default="euler")
    parser.add_argument("--sampling_num_steps", type=int, default=50)


    # --- TODO: remove ---
    # parser.add_argument("--config", type=str, help="Optional") # config loaded from experiment_path/config.yaml
    # parser.add_argument("--ckpt_path", type=str, help="Path to the model checkpoint. If provided, overrides the config.")
    # --------------------
    parser.add_argument("--jsonl_path", type=str, default="captions/train.jsonl")
    parser.add_argument("--sample_dir", type=str, required=True)
    parser.add_argument("--per_proc_batch_size", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--fid_stat_npz", type=str, required=True, help="Path to the .npz file containing the precomputed FID statistics.")
    parser.add_argument("--make_npz", action='store_true', default=False, help="Whether to create a .npz file from the sampled images.")
    parser.add_argument("--remove_temp_dir", type=bool, default=True, help="Whether to remove the temporary sample directory after creating the .npz file.")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--cfg_method", type=str, default=None, choices=[None, "cfg", "ag"], help="Use classifier-free guidance during sampling.")
    parser.add_argument("--cfg_scale", type=float, default=2.0, help="CFG scale if --cfg is set.")
    parser.add_argument("--cfg_interval", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable TF32 matmuls (Ampere+). Disable if deterministic results are required.")
    parser.add_argument("--autoguidance_exp_config", type=str, default=None, help="Path to autoguidance model config yaml.")
    parser.add_argument("--autoguidance_ckpt", type=str, default=None, help="Path to autoguidance model checkpoint.")
    parser.add_argument("--autoguidance_train_step", type=int, default=None, help="Train step for autoguidance model checkpoint if --autoguidance_ckpt is not provided.")
    args = parser.parse_args()
    main(args)
    