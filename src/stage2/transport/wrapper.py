import torch
from . import path
# from .transport_v2 import PathType

class Wrapper_v2(torch.nn.Module):
    def __init__(self, dit, textve, path_type):
        super().__init__()
        
        path_options = {
            "Linear": path.ICPlan,
            "GVP": path.GVPCPlan,
            "VP": path.VPCPlan,
        }
        
        self.dit = dit
        self.textve = textve
        self.path_sampler = path_options[path_type]()
    
    # def forward(self, x, t = None, mask=None, textve=False, **kwargs):
    #     if textve:
    #         return self.textve(text_tokens=x, text_key_padding_mask=mask)
    #     else:
    #         return self.dit(x, t, **kwargs)
    
    def forward(self, x1, t, clip_latents, padding_masks, **kwargs):
        x0, mu, log_var = self.textve(text_tokens=clip_latents, text_key_padding_mask=padding_masks)
        
        x0_2d = x0.permute(0, 2, 1).contiguous().view(x1.shape)
        t, xt, ut = self.path_sampler.plan(t, x0_2d, x1)
        
        model_output = self.dit(xt, t, **kwargs)
        
        return model_output, xt, ut, x0, mu, log_var