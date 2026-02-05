import numpy as np
import torch as th
import torch.nn as nn
from torchdiffeq import odeint
from functools import partial
from tqdm import tqdm

class sde:
    """SDE solver class"""
    def __init__(
        self, 
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
        time_dist_shift,
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        self.t = 1 - th.linspace(t0, t1, num_steps)
        self.t = time_dist_shift * self.t / (1 + (time_dist_shift - 1) * self.t)
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type
        self.time_dist_shift = time_dist_shift

    def __Euler_Maruyama_step(self, x, mean_x, t_curr, t_next, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t_curr
        dw = w_cur * th.sqrt(t_curr - t_next)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x - drift * (t_curr - t_next)
        x = mean_x + th.sqrt(2 * diffusion) * dw
        return x, mean_x
    
    def __Heun_step(self, x, _, t_curr, t_next, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(t_curr - t_next)
        diffusion = self.diffusion(x, th.ones(x.size(0)).to(x) * t_curr)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(
            xhat, th.ones(x.size(0)).to(x) * t_curr, model, **model_kwargs
        )
        xp = xhat - (t_curr - t_next) * K1
        K2 = self.drift(
            xp, th.ones(x.size(0)).to(x) * t_next, model, **model_kwargs
        )
        return xhat - 0.5 * (t_curr - t_next) * (K1 + K2), xhat # at last time point we do not perform the heun step

    def __forward_fn(self):
        """TODO: generalize here by adding all private functions ending with steps to it"""
        sampler_dict = {
            "euler": self.__Euler_Maruyama_step,
            "heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Smapler type not implemented.")
    
        return sampler

    def sample(self, init, model, **model_kwargs) -> tuple[th.Tensor]:
        """forward loop of sde"""
        x = init
        mean_x = init 
        samples = []
        sampler = self.__forward_fn()
        for t_curr, t_next in zip(self.t[:-1], self.t[1:]):
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, t_curr, t_next, model, **model_kwargs)
                samples.append(x)

        return samples

class ode:
    """ODE solver class"""
    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        time_dist_shift,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        # self.t = th.linspace(t0, t1, num_steps)
        self.t = 1 - th.linspace(t0, t1, num_steps)
        self.t = time_dist_shift * self.t / (1 + (time_dist_shift - 1) * self.t)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs) -> tuple[th.Tensor]:
        
        device = x[0].device if isinstance(x, tuple) else x.device
        def _fn(t, x):
            t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
            model_output = self.drift(x, t, model, **model_kwargs)
            return model_output

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(
            _fn,
            x,
            t,
            method=self.sampler_type,
            atol=atol,
            rtol=rtol
        )
        return samples

class ode_x:
    """ODE solver class"""
    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        time_dist_shift,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        # self.t = th.linspace(t0, t1, num_steps)
        self.t = 1 - th.linspace(t0, t1, num_steps)
        self.t = time_dist_shift * self.t / (1 + (time_dist_shift - 1) * self.t)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs) -> tuple[th.Tensor]:
        
        device = x[0].device if isinstance(x, tuple) else x.device
        def _fn(t, x):
            t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
            model_output = self.drift(x, t, model, **model_kwargs)
            return model_output

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(
            _fn,
            x,
            t,
            method=self.sampler_type,
            atol=atol,
            rtol=rtol
        )
        return samples
    
class ode_npe:
    """ODE solver class"""
    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        time_dist_shift,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        # self.t = th.linspace(t0, t1, num_steps)
        self.t = 1 - th.linspace(t0, t1, num_steps)
        self.t = time_dist_shift * self.t / (1 + (time_dist_shift - 1) * self.t)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs) -> tuple[th.Tensor]:
        # 기존 함수 유지
        device = x[0].device if isinstance(x, tuple) else x.device
        def _fn(t, x):
            t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
            model_output = self.drift(x, t, model, **model_kwargs)
            return model_output

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(
            _fn,
            x,
            t,
            method=self.sampler_type,
            atol=atol,
            rtol=rtol
        )
        return samples

    # [NEW] NPE 측정을 위한 함수 추가
    def sample_with_energy(self, x, model, **model_kwargs):
        """
        Samples and calculates Path Energy (integral of velocity squared).
        Returns:
            samples: Trajectory of x (T, B, ...)
            final_energy: Total energy consumed for each batch item (B,)
        """
        device = x.device
        
        # Augmented ODE function: State is (x, energy)
        def _fn(t, state):
            x_curr, _ = state # energy는 미분값 계산에 필요 없으므로 무시
            
            # t broadcasting
            t_vec = th.ones(x_curr.size(0)).to(device) * t
            
            # 1. Calculate Velocity (v)
            v = self.drift(x_curr, t_vec, model, **model_kwargs)
            
            # 2. Calculate Energy Derivative (dE/dt = |v|^2)
            # flatten(1) -> (B, D) -> sum(dim=1) -> (B,)
            dE_dt = v.pow(2).flatten(1).sum(dim=1)
            
            return (v, dE_dt)

        t = self.t.to(device)
        
        # Initial State: (x_start, energy_start=0)
        e0 = th.zeros(x.size(0)).to(device)
        initial_state = (x, e0)
        
        atol = [self.atol, self.atol] # x와 energy 각각에 대한 허용 오차
        rtol = [self.rtol, self.rtol]
        
        # Solve Augmented ODE
        # results will be a tuple: (x_traj, energy_traj)
        # x_traj shape: (Time, Batch, ...)
        # energy_traj shape: (Time, Batch)
        results = odeint(
            _fn,
            initial_state,
            t,
            method=self.sampler_type,
            atol=atol,
            rtol=rtol
        )
        
        x_traj, energy_traj = results
        
        # Return trajectory of x, and the FINAL accumulated energy
        return x_traj, energy_traj[-1]