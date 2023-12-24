import torch
import copy


class EMA:
    def __init__(self, model, base_gamma, total_steps):
        super().__init__()
        self.online_model = model

        self.ema_model = copy.deepcopy(self.online_model)
        self.ema_model.requires_grad_(False)

        self.base_gamma = base_gamma
        self.total_steps = total_steps

    def update_params(self, gamma):
        with torch.no_grad():
            valid_types = [torch.float, torch.float16]
            for o_param, t_param in self._get_params():
                if o_param.dtype in valid_types and t_param.dtype in valid_types:
                    t_param.data.lerp_(o_param.data, 1. - gamma)

            for o_buffer, t_buffer in self._get_buffers():
                if o_buffer.dtype in valid_types and t_buffer.dtype in valid_types:
                    t_buffer.data.lerp_(o_buffer.data, 1. - gamma)

    def _get_params(self):
        return zip(self.online_model.parameters(),
                   self.ema_model.parameters())

    def _get_buffers(self):
        return zip(self.online_model.buffers(),
                   self.ema_model.buffers())
    
    # cosine EMA schedule (increase from base_gamma to 1)
    # k -> current training step, K -> maximum number of training steps
    def update_gamma(self, current_step):
        k = torch.tensor(current_step, dtype=torch.float32)
        K = torch.tensor(self.total_steps, dtype=torch.float32)

        tau = 1 - (1 - self.base_gamma) * (torch.cos(torch.pi * k / K) + 1) / 2
        return tau.item()
