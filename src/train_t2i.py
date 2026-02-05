# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import os
import torch
from torch import autocast
import subprocess

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid

import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import json

import math
from omegaconf import OmegaConf
from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import ModelType, create_transport, Sampler
from utils.train_utils import parse_configs
from utils.model_utils import instantiate_from_config
from utils import wandb_utils
import wandb
from utils.optim_utils import build_optimizer, build_scheduler
from utils.data_utils import IN1KTextJsonlDataset, center_crop_arr
from stage2.text_encoders.perceiver import PerceiverVE
from stage2.text_encoders.clip import FrozenCLIPEmbedder
from stage2.text_encoders.contrastive_loss import ClipLoss, SigLipLoss
from stage2.text_encoders.regularization_loss import *
from stage2 import Wrapper

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr_224(x):
    return center_crop_arr(x, 224)

def center_crop_arr_256(x):
    return center_crop_arr(x, 256)

def center_crop_arr_512(x):
    return center_crop_arr(x, 512)

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(rank, args):
    """Trains a new SiT model using config-driven hyperparameters."""
    use_xla = args.use_xla
    if use_xla:
        import torch_xla as xla
        import torch_xla.core.xla_model as xm
        from torch_xla.distributed.parallel_loader import ParallelLoader
        device_type = 'xla'
    else:
        device_type = 'cuda'
    (
        rae_config,
        model_config,
        textve_config,
        clip_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        training_config,
    ) = parse_configs(args.config)

    if rae_config is None or model_config is None:
        raise ValueError("Config must provide both stage_1 and stage_2 sections.")

    def to_dict(cfg_section):
        if cfg_section is None:
            return {}
        return OmegaConf.to_container(cfg_section, resolve=True)

    misc = to_dict(misc_config)
    transport_cfg = to_dict(transport_config)
    sampler_cfg = to_dict(sampler_config)
    guidance_cfg = to_dict(guidance_config)
    training_cfg = to_dict(training_config)

    num_classes = int(misc.get("num_classes", 1000)) # Legacy
    null_label = int(misc.get("null_label", num_classes))
    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    
    detach_ut = misc.get("diffusion_loss_detach_ut", True)
    use_kld_loss = misc.get("use_kld_loss", False)
    use_align_loss = misc.get("use_align_loss", False)
    
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    clip_grad = float(training_cfg.get("clip_grad", 1.0))
    clip_logvar_min = float(training_cfg.get("clip_logvar_min", -10.0))
    clip_logvar_max = float(training_cfg.get("clip_logvar_max", 10.0))
    clip_logit_scale = float(training_cfg.get("clip_logit_scale", -1))
    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    epochs = int(training_cfg.get("epochs", 1400))
    global_batch_size = int(training_cfg.get("global_batch_size", 1024))
    num_workers = int(training_cfg.get("num_workers", 4))
    log_every = int(training_cfg.get("log_every", 100))
    ckpt_every = int(training_cfg.get("ckpt_every", 5_000))
    sample_every = int(training_cfg.get("sample_every", 10_000))
    cfg_scale_override = training_cfg.get("cfg_scale", None)
    default_seed = int(training_cfg.get("global_seed", 0))
    global_seed = args.global_seed if args.global_seed is not None else default_seed

    if grad_accum_steps < 1:
        raise ValueError("Gradient accumulation steps must be >= 1.")
    if args.image_size % 16 != 0:
        raise ValueError("Image size must be divisible by 16 for the RAE encoder.")

    if use_xla:
        dist.init_process_group("xla", init_method='xla://')
    else:
        dist.init_process_group("nccl")
    
    world_size = dist.get_world_size()
    if global_batch_size % (world_size * grad_accum_steps) != 0:
        raise ValueError("Global batch size must be divisible by world_size * grad_accum_steps.")
    if rank is None:
        rank = dist.get_rank()
    local_rank = xm.get_local_ordinal() if use_xla else int(os.environ["LOCAL_RANK"])
    
    if use_xla:
        device = xla.device()
    else:
        device_idx = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_idx)
        device = torch.device("cuda", device_idx)

    seed = global_seed * world_size + rank

    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_xla:
        xm.set_rng_state(seed) # TODO
    else:
        torch.cuda.manual_seed(seed)
    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    

    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)
    use_bf16 = args.precision == "bf16"

    autocast_kwargs = dict(device_type = device_type, dtype=torch.bfloat16, enabled=use_bf16)
    latent_dtype = autocast_kwargs["dtype"] if use_bf16 else torch.float32

    transport_params = dict(transport_cfg.get("params", {}))
    path_type = transport_params.get("path_type", "Linear")
    prediction = transport_params.get("prediction", "velocity")
    loss_weight = transport_params.get("loss_weight")
    transport_params.pop("time_dist_shift", None)

    sampler_mode = sampler_cfg.get("mode", "ODE").upper()
    sampler_params = dict(sampler_cfg.get("params", {}))

    guidance_scale = float(guidance_cfg.get("scale", 1.0))
    if cfg_scale_override is not None:
        guidance_scale = float(cfg_scale_override)
    guidance_method = guidance_cfg.get("method", "cfg")

    def guidance_value(key: str, default: float) -> float:
        if key in guidance_cfg:
            return guidance_cfg[key]
        dashed_key = key.replace("_", "-")
        return guidance_cfg.get(dashed_key, default)

    t_min = float(guidance_value("t_min", 0.0))
    t_max = float(guidance_value("t_max", 1.0))

    if args.exp_name is not None:
        experiment_name = f"{args.exp_name}"
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%m%d%H%M")
        experiment_index = len(glob(f"{args.exps_dir}/*")) - 1
        model_target = str(model_config.get("target", "stage2"))
        model_string_name = model_target.split(".")[-1]
        precision_suffix = f"-{args.precision}" if args.precision == "bf16" else ""
        loss_weight_str = loss_weight if loss_weight is not None else "none"
        experiment_name = (
            f"{experiment_index:03d}-{timestamp}-{model_string_name}-"
            f"{path_type}-{prediction}-{loss_weight_str}{precision_suffix}-acc{grad_accum_steps}"
        )
    experiment_dir = os.path.join(args.exps_dir, experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    
    if rank == 0:
        os.makedirs(args.exps_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        
        # save config file
        config_path = os.path.join(experiment_dir, "config.yaml")
        if not os.path.exists(config_path): # 1) new exp
            OmegaConf.save(OmegaConf.load(args.config), config_path)
        else: # 2) existing exp (config file exists)
            logger.info(f"Config file already exists at {config_path}.")
            existing_config = OmegaConf.load(config_path)
            new_config = OmegaConf.load(args.config)
            if existing_config == new_config: # 2-1) config matches -> OK
                logger.info(f"Existing config and new config match! Train can proceed.")
            else: # 2-2) config mismatch -> OK only when no ckpt exists
                try:
                    assert len(os.listdir(checkpoint_dir)) == 0, "Cannot overwrite config file when checkpoints exist."
                    logger.info(f"Existing config and new config do not match! But there are no checkpoints. Overwriting existing config file.")
                except:
                    logger.error(f"[WARNING] Existing config and new config do not match! Check whether this is expected before proceeding.")
                OmegaConf.save(OmegaConf.load(args.config), config_path)
                

        if args.wandb:
            entity = os.environ["ENTITY"]
            project = os.environ["PROJECT"]
            wandb_utils.initialize(args, entity, experiment_name, project, config_yaml=OmegaConf.load(args.config))
    else:
        logger = create_logger(None)

    rae: RAE = instantiate_from_config(rae_config)
    model_class = rae_config.get("target", "").lower()
    
    rae.to(device)
    if args.rae_decoder_to_cpu:
        rae.decoder.to("cpu")
    rae.eval()
    
    clip_config['params']['device'] = device_type
    clip: FrozenCLIPEmbedder = instantiate_from_config(clip_config).to(device)
    clip.eval()

    dit: Stage2ModelProtocol = instantiate_from_config(model_config)
    wrapping_modules ={"dit": dit}
    textve = None
    if textve_config is not None:
        if textve_config.get("target", None) is not None:
            textve = instantiate_from_config(textve_config).to(device)
            wrapping_modules["textve"] = textve
    
    align_logit_scale = None
    if use_align_loss:
        align_loss_weight = misc.get("align_loss_weight", 0.1)
        align_loss_type = misc.get("align_loss_type", "l2")
        if align_loss_type in ["siglip", "clip"]:
            align_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1/0.07))
            wrapping_modules["align_logit_scale"] = align_logit_scale
        
    wrapper_model = Wrapper(**wrapping_modules).to(device)
    ema = deepcopy(wrapper_model).to(device)

    opt_state = None
    sched_state = None
    train_steps = 0
    
    if args.initialize_from is not None:
        assert args.ckpt is None
        assert args.resume_last is False
        checkpoint = torch.load(args.initialize_from, map_location="cpu")
        if "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"], strict=False)
            wrapper_model.load_state_dict(checkpoint["ema"], strict=False)
            logger.info(f"Initializing model...")
        else:
            raise ValueError("No EMA weights found in the specified checkpoint for initialization.")
        start_steps = int(checkpoint.get("train_steps", 0))
        
        del checkpoint
        import gc
        gc.collect()
        if not use_xla: torch.cuda.empty_cache()
        logger.info(f"Initialized checkpoint from {os.path.basename(args.initialize_from)} with step {start_steps}")

    if args.ckpt is not None:
        assert args.resume_last is False
        assert args.initialize_from is None
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        if "model" in checkpoint:
            wrapper_model.load_state_dict(checkpoint["model"], strict=False)
            logger.info(f"Loading model...")
        if "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"], strict=False)
            logger.info(f"Loading EMA model...")
                
        opt_state = checkpoint.get("opt")
        sched_state = checkpoint.get("scheduler")
        train_steps = int(checkpoint.get("train_steps", 0))

        del checkpoint
        import gc
        gc.collect()
        if not use_xla: torch.cuda.empty_cache()

        logger.info(f"Loaded checkpoint from {os.path.basename(args.ckpt)} with step {train_steps}")

    if args.resume_last:
        assert args.ckpt is None
        assert args.initialize_from is None
        last_ckpt_list = glob(f"{checkpoint_dir}/last-*.pt")
        if len(last_ckpt_list) == 0:
            logger.warning(f"No last checkpoint found in {checkpoint_dir}. Try downloading from remote storage.")
        elif len(last_ckpt_list) > 1:
            logger.warning(f"Expected exactly one last checkpoint in {checkpoint_dir}, found {len(last_ckpt_list)}.")
        else:
            last_ckpt_path = max(last_ckpt_list, key=lambda x: int(x.split('last-')[-1].replace('.pt', '')))
            checkpoint = torch.load(last_ckpt_path, map_location="cpu")
            if "model" in checkpoint:
                wrapper_model.load_state_dict(checkpoint["model"])
                logger.info(f"Loading model...")
            if "ema" in checkpoint:
                ema.load_state_dict(checkpoint["ema"])
                logger.info(f"Loading EMA model...")
                    
            opt_state = checkpoint.get("opt")
            sched_state = checkpoint.get("scheduler")
            train_steps = int(checkpoint.get("train_steps", 0))

            del checkpoint
            import gc
            gc.collect()
            if not use_xla:  torch.cuda.empty_cache()

            logger.info(f"Loaded last checkpoint from {os.path.basename(last_ckpt_path)} with step {train_steps}")

    dit_param_count = sum(p.numel() for p in dit.parameters())
    logger.info(f"DiTDH Parameters: {dit_param_count/1e6:.2f}M")
    
    if textve is not None:
        textve_param_count = sum(p.numel() for p in textve.parameters())
        logger.info(f"TextVE Parameters: {textve_param_count/1e6:.2f}M")
    
    total_param_count = sum(p.numel() for p in wrapper_model.parameters())
    logger.info(f"Total Stage 2 Parameters: {total_param_count/1e6:.2f}M")

    clip_param_count = sum(p.numel() for p in clip.parameters())
    rae_param_count = sum(p.numel() for p in rae.parameters())
    logger.info(f"Frozen Parameters -- CLIP: {clip_param_count/1e6:.2f}M, RAE: {rae_param_count/1e6:.2f}M")
    

    if use_xla:
        xm.broadcast_master_param(wrapper_model)
        wrapper_model_woddp = wrapper_model
        xm.mark_step()
    else:
        find_unused_parameters = False
        wrapper_model = DDP(wrapper_model, device_ids=[device_idx], gradient_as_bucket_view=False, find_unused_parameters=find_unused_parameters) # For temporary test purpose, should modify forward of wrapper.
        wrapper_model_woddp = wrapper_model.module
    
    if use_kld_loss:
        assert textve is not None, "KLD loss requires a textve model."
        kld_loss_weight = misc.get("kld_loss_weight", 1e-2)
        kld_reduction = misc.get("kld_reduction", "mean")
        kld_loss_type = misc.get("kld_loss_type", "naive_kld")
        kld_target_std = misc.get("kld_target_std", 1.0)

    if use_align_loss:
        assert textve is not None, "Alignment loss requires a textve model."
        if align_loss_type in ["siglip", "clip"]:
            if align_loss_type == "siglip":
                dist_impl = "gather" if use_xla else None
                align_loss_fn = SigLipLoss(rank=rank, world_size=world_size, dist_impl=dist_impl)
            elif align_loss_type == "clip":
                align_loss_fn = ClipLoss(rank=rank, world_size=world_size, gather_with_grad=True, local_loss=False, cache_labels=True, is_xla=use_xla)
        else:
            def align_loss_fn(x, y, normalize=False):
                x = F.normalize(x, p=2, dim=-1) if normalize else x
                y = F.normalize(y, p=2, dim=-1) if normalize else y
                return F.mse_loss(x, y)
                
    opt, opt_msg = build_optimizer(wrapper_model.parameters(), training_cfg)
    if opt_state is not None:
        opt.load_state_dict(opt_state)


    # local_fn(fn inside main()) not supported on xla/tpu
    if args.image_size == 256:
        cener_crop_fn = center_crop_arr_256
    elif args.image_size == 512:
        cener_crop_fn = center_crop_arr_512
    elif args.image_size == 224:
        cener_crop_fn = center_crop_arr_224
    else:
        raise NotImplementedError(f"Center crop function not implemented for image size {args.image_size}.")

    transform = transforms.Compose([
        transforms.Lambda(cener_crop_fn),
        transforms.ToTensor(),
    ])
    
    dataset = IN1KTextJsonlDataset(
        jsonl_path=os.path.join(args.jsonl_path),
        image_root=args.data_path,
        resolution=args.image_size,
        transform=transform,
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False if use_xla else training_cfg.get("pin_memory", True),
        drop_last=True,
        prefetch_factor= training_cfg.get("prefetch_factor", None),
        persistent_workers=training_cfg.get("persistent_workers", False),
    )

    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    logger.info(
        f"Gradient accumulation: steps={grad_accum_steps}, micro batch={micro_batch_size}, "
        f"per-GPU batch={micro_batch_size * grad_accum_steps}, global batch={global_batch_size}"
    )
    logger.info(f"Precision mode: {args.precision}")

    loader_batches = len(loader)
    if loader_batches % grad_accum_steps != 0:
        raise ValueError("Number of loader batches must be divisible by grad_accum_steps when drop_last=True.")
    steps_per_epoch = loader_batches // grad_accum_steps
    if steps_per_epoch <= 0:
        raise ValueError("Gradient accumulation configuration results in zero optimizer steps per epoch.")
    schedl, sched_msg = build_scheduler(opt, steps_per_epoch, training_cfg, sched_state)
    if rank == 0:
        logger.info(f"Training configured for {epochs} epochs, {steps_per_epoch} steps per epoch.")
        logger.info(opt_msg + "\n" + sched_msg)
        
    transport = create_transport(
        **transport_params,
        time_dist_shift=time_dist_shift,
    )
    transport_sampler = Sampler(transport)
    assert transport.model_type == ModelType.VELOCITY, "Only velocity model is supported in this script."

    if sampler_mode == "ODE":
        eval_sampler = transport_sampler.sample_ode_npe(**sampler_params)
    elif sampler_mode == "SDE":
        eval_sampler = transport_sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")

    guid_model_forward = None
    if guidance_scale > 1.0 and guidance_method == "autoguidance":
        guidance_model_cfg = guidance_cfg.get("guidance_model")
        if guidance_model_cfg is None:
            raise ValueError("Please provide a guidance model config when using autoguidance.")
        guid_model: Stage2ModelProtocol = instantiate_from_config(guidance_model_cfg).to(device)
        guid_model.eval()
        guid_model_forward = guid_model.forward
    
    update_ema(ema, wrapper_model_woddp, decay=0)
    wrapper_model.train()
    ema.eval()

    log_dict = {
        "train_loss": {"running": torch.tensor(0.0, device=device), "step_accum":torch.tensor(0.0, device=device) },
        "diffusion_loss": {"running": torch.tensor(0.0, device=device), "step_accum": torch.tensor(0.0, device=device) },
    }
    if textve is not None: 
        log_dict["x0_norm"] = {"running": torch.tensor(0.0, device=device), "step_accum": torch.tensor(0.0, device=device) }
        log_dict["x0_std"] = {"running": torch.tensor(0.0, device=device), "step_accum": torch.tensor(0.0, device=device) }
        log_dict["textve_mu_norm"] = {"running": torch.tensor(0.0, device=device), "step_accum": torch.tensor(0.0, device=device) }
        log_dict["textve_mu_std"] = {"running": torch.tensor(0.0, device=device), "step_accum": torch.tensor(0.0, device=device) }
        log_dict["textve_std_norm"] = {"running": torch.tensor(0.0, device=device), "step_accum": torch.tensor(0.0, device=device) }
    
    if use_kld_loss:
        log_dict["kld_loss"] = {"running": torch.tensor(0.0, device=device), "step_accum": torch.tensor(0.0, device=device) }
    if use_align_loss:
        log_dict["align_loss"] = {"running": torch.tensor(0.0, device=device), "step_accum": torch.tensor(0.0, device=device) }
        if align_loss_type in ["siglip", "clip"]:
            log_dict["align_logit_scale"] = {"running": torch.tensor(0.0, device=device), "step_accum": torch.tensor(0.0, device=device) }
    
    log_steps = 0
    start_time = time()
    
    # if rank == 0:
    with open(args.jsonl_path_val, "r") as f:
        val_captions = json.load(f)
    
    max_len = min(micro_batch_size, 64)
    val_captions = val_captions[:max_len]
    if use_xla:
        @torch.no_grad()
        def model_fn(*args, **kwargs):
            out = ema.dit.forward(*args, **kwargs)
            # xm.mark_step() # use only when oom
            return out
    else:
        model_fn = ema.dit.forward
    
    with torch.no_grad():
        
        val_clip_latents, val_latents_and_others = clip.encode(val_captions)
        val_valid_masks = val_latents_and_others["token_mask"]
        val_clip_padding_masks = ~val_valid_masks.to(torch.bool)
        val_clip_pooler_output = val_latents_and_others["pooler_output"]
        
        null_clip_latents, null_latents_and_others = clip.encode("")
        null_valid_masks = null_latents_and_others["token_mask"]
        
    if textve is None:
        # noise source
        sample_x0 = torch.randn(max_len, *latent_size, device=device, dtype=latent_dtype)
    else:
        # learned source
        sample_textve_kwargs = dict(
            text_tokens=val_clip_latents,
            text_key_padding_mask=val_clip_padding_masks,
        )
    sample_model_kwargs = dict(
        y=val_clip_pooler_output,
        context=val_clip_latents,
        context_mask=val_valid_masks.to(torch.bool),
        null_context=null_clip_latents,
        null_context_mask=null_valid_masks.to(torch.bool),
    )
    
    logger.info(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        epoch_loader = loader if not use_xla else ParallelLoader(loader, [device]).per_device_loader(device)
        logger.info(f"Beginning epoch {epoch}...")
        opt.zero_grad()
        accum_counter = 0

        for key in log_dict.keys():
            log_dict[key]["step_accum"] = 0.0
            
        
        for x, captions in epoch_loader: # img, text
            x = x.to(device)
            with torch.no_grad():
                x = rae.encode(x)
            x1 = x
            
            with torch.no_grad():
                clip_latents, latents_and_others = clip.encode(captions)
                clip_valid_masks = latents_and_others["token_mask"]
                clip_padding_masks = ~clip_valid_masks.to(torch.bool)
                clip_pooler_output = latents_and_others["pooler_output"]
                
            with autocast(**autocast_kwargs):
                # 1) x0
                if textve is None: # noise
                    x0 = torch.randn_like(x1)
                else: # learned
                    textve_kwargs = {"text_tokens": clip_latents, "text_key_padding_mask": clip_padding_masks}
                    x0_token, mu, log_var = wrapper_model('textve', **textve_kwargs) # B, S, D
                    
                    if log_var is None: # use variational=false for non klds
                        std = torch.full_like(x0_token, misc.get("std", 0.1))
                        eps = torch.randn_like(x0_token)
                        x0_token = mu + eps * std
                    else:
                        log_var = torch.clamp(log_var, min=clip_logvar_min, max=clip_logvar_max)
                    x0 = x0_token.permute(0, 2, 1).contiguous().view(x1.shape) # B, D, S
                    # logging stats
                    textve_mu_norm = mu.flatten(1).norm(dim=1).mean()
                    textve_mu_std = mu.flatten(1).std(dim=0).mean()
                    textve_std_norm = torch.exp(0.5 * log_var).flatten(1).norm(dim=1).mean() if log_var is not None else torch.tensor(0.0, device=device)
                    
                    x0_norm = x0.flatten(1).norm(dim=1).mean()
                    x0_std = x0.flatten(1).std(dim=0).mean()
                    
                    log_dict["x0_norm"]['step_accum'] += x0_norm.detach()
                    log_dict["x0_std"]['step_accum'] += x0_std.detach()
                    log_dict["textve_mu_norm"]['step_accum'] += textve_mu_norm.detach()
                    log_dict["textve_mu_std"]['step_accum'] += textve_mu_std.detach()
                    log_dict["textve_std_norm"]['step_accum'] += textve_std_norm.detach()
                        
                # 2) xt, ut, forwarding
                t = transport.sample_timestep(x1)
                t, xt, ut = transport.path_sampler.plan(t, x0, x1)

                dit_kwargs = {"x": xt, "t": t, "y": clip_pooler_output, "context": clip_latents, "context_mask": clip_valid_masks.to(torch.bool), "null_context": null_clip_latents, "null_context_mask": null_valid_masks.to(torch.bool)}
                model_output = wrapper_model('dit', **dit_kwargs)

                # 4) loss
                ut_gt = ut.detach() if detach_ut else ut
                diffusion_loss = ((model_output - ut_gt) ** 2).mean()
                log_dict["diffusion_loss"]['step_accum'] += diffusion_loss.detach()
                loss_tensor = diffusion_loss
                
                if use_kld_loss:
                    kld_loss = kld_loss_factory(mu, log_var, kld_loss_type, reduction=kld_reduction, kld_target_std=kld_target_std)
                    loss_tensor = loss_tensor + kld_loss_weight * kld_loss
                    log_dict['kld_loss']['step_accum'] += kld_loss.detach()
                
                if use_align_loss:
                    if align_loss_type in ["siglip", "clip"]:
                        align_logit_scale = wrapper_model('align_logit_scale')
                        if misc.get("align_loss_w_mean_token", True):
                            anchor = mu.mean(dim=1)
                            target = x1.mean(dim=(-1,-2))
                        else:
                            x1_token = x1.flatten(2).permute(0, 2, 1) # [B C H W -> B HW C]
                            anchor = mu.flatten(1).contiguous()
                            target = x1_token.flatten(1).contiguous()

                        anchor, target = anchor / anchor.norm(dim=-1, keepdim=True), target / target.norm(dim=-1, keepdim=True)
                        align_loss = align_loss_fn(anchor, target, logit_scale=align_logit_scale.exp())
                        loss_tensor = loss_tensor + align_loss_weight * align_loss
                        log_dict["align_logit_scale"]['step_accum'] += align_logit_scale.detach()
                    elif align_loss_type in ["l2", "normalized_l2"]:
                        mu_reshaped = mu.permute(0, 2, 1).contiguous().view(x1.shape)
                        normalize = align_loss_type == "normalized_l2"
                        align_loss = align_loss_fn(mu_reshaped, x1, normalize=normalize)
                        loss_tensor = loss_tensor + align_loss_weight * align_loss
                    elif align_loss_type == "norm":
                        mu_reshaped = mu.permute(0, 2, 1).contiguous().view(x1.shape)
                        mu_norm = mu_reshaped.norm(dim=1)
                        x1_norm = x1.norm(dim=1)
                        align_loss = F.mse_loss(mu_norm, x1_norm)
                        loss_tensor = loss_tensor + align_loss_weight * align_loss
                        
                    log_dict['align_loss']['step_accum'] += align_loss.detach()
                        
            log_dict["train_loss"]['step_accum'] += loss_tensor.detach()
            (loss_tensor / grad_accum_steps).backward()
            accum_counter += 1

            if accum_counter < grad_accum_steps:
                continue

            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(wrapper_model.parameters(), clip_grad)
            opt.step() if not use_xla else xm.optimizer_step(opt)
            
            if use_align_loss and align_loss_type in ["siglip", "clip"] and clip_logit_scale > 0:
                with torch.no_grad():
                    wrapper_model_woddp.align_logit_scale.clamp_(min=0.0, max=clip_logit_scale)
                    
            
            schedl.step()
            update_ema(ema, wrapper_model_woddp, decay=ema_decay)
            opt.zero_grad(set_to_none=True)

            for key in log_dict.keys():
                log_dict[key]['running'] += log_dict[key]['step_accum'] / grad_accum_steps
                log_dict[key]['step_accum'] = torch.tensor(0.0, device=device)
            
            log_steps += 1
            train_steps += 1
            accum_counter = 0
            
            if train_steps % log_every == 0:
                torch.cuda.synchronize() if not use_xla else xm.mark_step()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_dict = {}
                for key in log_dict.keys():
                    avg_value = log_dict[key]['running'] / log_steps
                    dist.all_reduce(avg_value, op=dist.ReduceOp.SUM)
                    avg_value = avg_value.item() / world_size
                    avg_dict[key] = avg_value

                msg = f"(step={train_steps:07d}) "+ \
                        ", ".join([f"{key}: {avg_dict[key]:.4f}" for key in avg_dict.keys()]) + \
                        f", train_steps/Sec: {steps_per_sec:.2f}"
                logger.info(msg)

                if args.wandb:
                    avg_dict.update({
                        "train_steps/sec": steps_per_sec,
                        "learning_rate": schedl.get_last_lr()[0],
                    })
                    wandb_utils.log(
                        avg_dict,
                        step=train_steps,
                    )

                for key in log_dict.keys():
                    log_dict[key]['running'] = torch.tensor(0.0, device=device)
                
                log_steps = 0
                start_time = time()

            if train_steps % ckpt_every == 0 and train_steps > 0:
                if use_xla: xm.mark_step()
                if rank == 0:
                    checkpoint = {
                        "model": wrapper_model_woddp.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": schedl.state_dict(),
                        "train_steps": train_steps,
                        "config_path": args.config,
                        "training_cfg": training_cfg,
                        "cli_overrides": {
                            "data_path": args.data_path,
                            "exps_dir": args.exps_dir,
                            "image_size": args.image_size,
                            "precision": args.precision,
                            "global_seed": global_seed,
                        },
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier() if not use_xla else xm.rendezvous("ckpt_done barrier")

            if train_steps % args.last_ckpt_every == 0 and train_steps > 0:
                if use_xla: xm.mark_step()
                if rank == 0:
                    checkpoint = {
                        "model": wrapper_model_woddp.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": schedl.state_dict(),
                        "train_steps": train_steps,
                        "config_path": args.config,
                        "training_cfg": training_cfg,
                        "cli_overrides": {
                            "data_path": args.data_path,
                            "exps_dir": args.exps_dir,
                            "image_size": args.image_size,
                            "precision": args.precision,
                            "global_seed": global_seed,
                        },
                    }
                    checkpoint_path = f"{checkpoint_dir}/last-{train_steps}.pt"
                    for fpath in glob(f"{checkpoint_dir}/last-*.pt"):
                        os.remove(fpath)
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved last checkpoint to {checkpoint_path}")                            
                dist.barrier() if not use_xla else xm.rendezvous("last_ckpt_done barrier")

            if train_steps % sample_every == 0 or (train_steps == 1 and args.sample_at_first):
                if use_xla: 
                    xm.mark_step()
                    xm.rendezvous("pre_sample_barrier")
                logger.info("Generating EMA samples...")
                # if rank == 0: 
                if args.rae_decoder_to_cpu: rae.decoder.to(device)
                with torch.no_grad():
                    if textve is not None:
                        _sample_x0, _, _ = ema.textve(**sample_textve_kwargs)
                        _sample_x0 = _sample_x0.permute(0, 2, 1).contiguous().view(max_len, *latent_size)
                    else:
                        _sample_x0 = sample_x0.clone()
                    
                    with autocast(**autocast_kwargs):
                        traj, path_energy = eval_sampler(_sample_x0, model_fn, **sample_model_kwargs)
                    samples = traj[-1]
                    straight_energy = (samples - _sample_x0).pow(2).flatten(1).sum(dim=1)
                    samples = rae.decode(samples.to(torch.float32))
                    
                    npe_batch = (path_energy.abs() - straight_energy).abs() / (straight_energy + 1e-8)
                    npe_mean = npe_batch.mean().item()
                    
                if args.rae_decoder_to_cpu: rae.decoder.to("cpu")
                logger.info(f"Generating EMA samples done. NPE: {npe_mean:.4f}")
                
                # if use_xla: xm.mark_step()
                if args.wandb and rank == 0:
                    _samples = make_grid(samples, nrow=round(math.sqrt(samples.size(0))), normalize=True, value_range=(0,1))
                    _samples = _samples.clamp(0, 1).mul(255).permute(1,2,0).to(torch.uint8)
                    _samples = _samples.to('cpu').numpy()
                    wandb.log({f"samples": wandb.Image(_samples)}, step=train_steps)
                    wandb.log({f"npe": npe_mean}, step=train_steps)
                    logger.info("Logging EMA samples done.")
                if use_xla: 
                    xm.mark_step()
                    xm.rendezvous("sample_done barrier")
                else:
                    dist.barrier()
                # dist.barrier() if not use_xla else xm.rendezvous("sample_done barrier")

        if accum_counter != 0:
            raise RuntimeError("Gradient accumulation counter not zero at epoch end.")

    wrapper_model.eval()
    logger.info("Done!")
    cleanup() if not use_xla else xm.rendezvous("cleanup")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to the JSONL file for the training dataset.")
    parser.add_argument("--jsonl_path_val", type=str, required=True, help="Path to the JSONL file for the validation dataset.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training dataset root.")
    parser.add_argument("--exps_dir", type=str, default="results", help="Directory to store training outputs.")
    parser.add_argument("--exp_name", type=str, default=None, help="Optional name for the experiment.")
    parser.add_argument("--image_size", type=int, choices=[224, 256, 512], default=256, help="Input image resolution.")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32", help="Compute precision for training.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to resume training.")
    parser.add_argument("--initialize_from", type=str, default=None, help="Optional checkpoint path to initialize training.")
    parser.add_argument("--global_seed", type=int, default=None, help="Override training.global_seed from the config.")
    parser.add_argument("--last_ckpt_every", type=int, default=5000, help="Save a checkpoint at the last step of every N steps.")
    parser.add_argument("--resume_last", action="store_true", help="Resume from the last checkpoint in the results directory.")
    parser.add_argument("--use_xla", action="store_true", help="Enable XLA training.")
    parser.add_argument("--xla_precision", type=str, choices=["default", "highest"], default="highest", help="XLA precision")
    parser.add_argument("--rae_decoder_to_cpu", action="store_true", help="Move RAE decoder to CPU to save GPU memory.")
    parser.add_argument("--sample_at_first", action="store_true", help="Generate samples at the first training step.")
    args = parser.parse_args()
    
    if args.use_xla:
        import torch_xla.distributed.xla_multiprocessing as xmp
        import torch_xla as xla
        xla._XLAC._xla_set_mat_mul_precision(args.xla_precision) # set precision to high to assure accuracy    
        xla.launch(main, args=(args,))
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        main(None, args)       