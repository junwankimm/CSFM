import torch.nn.functional as F
import torch as th
import math

def token_cosine_loss(x0: th.Tensor, x1: th.Tensor) -> th.Tensor:
    B, C, H, W = x0.shape
    assert x1.shape == x0.shape

    x0_flat = x0.view(B, C, H * W).permute(0, 2, 1)  # [B, N, C]
    x1_flat = x1.view(B, C, H * W).permute(0, 2, 1)  # [B, N, C]

    x0_flat = F.normalize(x0_flat, dim=-1)
    x1_flat = F.normalize(x1_flat, dim=-1)

    cos_sim = (x0_flat * x1_flat).sum(dim=-1)
    
    loss = -cos_sim.mean()
    return loss

def rkd_loss(x0: th.Tensor, x1: th.Tensor, pool_kernel: int = 2, pool_stride: int = None) -> th.Tensor:
    assert x0.shape == x1.shape, f"Shape mismatch: {x0.shape} vs {x1.shape}"
    B, C, H, W = x0.shape

    if pool_stride is None:
        pool_stride = pool_kernel

    if pool_kernel > 1:
        x0 = F.avg_pool2d(x0, kernel_size=pool_kernel, stride=pool_stride)
        x1 = F.avg_pool2d(x1, kernel_size=pool_kernel, stride=pool_stride)

    B, C, Hp, Wp = x0.shape
    N = Hp * Wp
    x0_flat = x0.flatten(2).transpose(1, 2)          # [B, N, C]
    x1_flat = x1.flatten(2).transpose(1, 2).detach() # [B, N, C]

    x0_flat = F.normalize(x0_flat, dim=-1)
    x1_flat = F.normalize(x1_flat, dim=-1)

    S0 = th.bmm(x0_flat, x0_flat.transpose(1, 2))
    S1 = th.bmm(x1_flat, x1_flat.transpose(1, 2))

    loss = F.mse_loss(S0, S1)
    return loss

def kld_loss_factory(mu, log_var, loss_type: str, reduction="mean", kld_target_std=1.0) -> th.Tensor:
    if loss_type == "naive_kld":
        return naive_kld_loss(mu, log_var, reduction=reduction, target_std=kld_target_std)
    elif loss_type == "kld":
        return standard_kld_loss(mu, log_var, reduction=reduction)
    elif loss_type == "var_kld":
        return var_only_kld_loss(mu, log_var, reduction=reduction, target_std=kld_target_std)
    else:
        raise ValueError(f"Unknown KLD loss type: {loss_type}")

# should be sum but dimension is too high -> we need weight of 1e-6 or something like that so just used mean
def naive_kld_loss(mu: th.Tensor, log_var: th.Tensor, reduction="mean", target_std=1.0) -> th.Tensor:
    var = log_var.exp()
    if target_std != 1.0:
        sigma2_star = target_std ** 2
        var = var / sigma2_star
        log_var = log_var - math.log(sigma2_star)
        
    if reduction == "mean":
        kld_loss = -0.5 * th.mean(1 + log_var - (0.3 * mu) ** 6 - var)
    elif reduction == "sum":
        kld_loss = -0.5 * (1 + log_var - (0.3 * mu) ** 6 - var).flatten(1).sum(1).mean()
    return kld_loss

def standard_kld_loss(mu: th.Tensor, log_var: th.Tensor, reduction="mean", target_std=1.0) -> th.Tensor:
    var = log_var.exp()
    if target_std != 1.0:
        sigma2_star = target_std ** 2
        var = var / sigma2_star
        log_var = log_var - math.log(sigma2_star)
    
    if reduction == "mean":
        kld_loss = -0.5 * th.mean(1 + log_var - mu ** 2 - var)
    elif reduction == "sum":
        kld_loss = -0.5 * (1 + log_var - mu ** 2 - var).flatten(1).sum(1).mean()
    return kld_loss

def var_only_kld_loss(mu: th.Tensor, log_var: th.Tensor, reduction="mean", target_std=1.0) -> th.Tensor:
    var = log_var.exp()
    if target_std != 1.0:
        sigma2_star = target_std ** 2
        var = var / sigma2_star
        log_var = log_var - math.log(sigma2_star)
        
    if reduction == "mean":
        kld_loss = -0.5 * th.mean(1 + log_var - var)
    elif reduction == "sum":
        kld_loss = -0.5 * (1 + log_var - var).flatten(1).sum(1).mean()
    return kld_loss

def legacy_var_only_kld_loss(mu: th.Tensor, log_var: th.Tensor) -> th.Tensor:
    kld_loss = -0.5 * th.sum(1 + log_var - log_var.exp(), dim=1).mean()
    return kld_loss