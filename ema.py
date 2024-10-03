# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION.
# All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
#cloner174
'''
Codes adapted and enhanced from https://github.com/NVlabs/LSGM/blob/main/util/ema.py
'''
import warnings
import torch
from torch.optim import Optimizer


class EMA:
    def __init__(self, model, optimizer, ema_decay=0.999, device=None):
        """
        Exponential Moving Average for model parameters.

        Args:
            model (torch.nn.Module): The model to apply EMA to.
            optimizer (torch.optim.Optimizer): The optimizer being used for training.
            ema_decay (float): Decay rate for EMA. Typical values are close to 1, e.g., 0.999.
            device (torch.device, optional): Device to store EMA parameters. If None, uses model's device.
        """
        self.ema_decay = ema_decay
        self.model = model
        self.optimizer = optimizer
        self.device = device if device is not None else next(model.parameters()).device
        self.apply_ema = self.ema_decay > 0.0

        if self.apply_ema:
            self.register_ema()

    def register_ema(self):
        """Initialize EMA parameters by cloning model parameters."""
        self.ema_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema_state[name] = param.detach().clone().to(self.device)
                self.ema_state[name].requires_grad = False  # EMA parameters should not require gradients

    @torch.no_grad()
    def step(self):
        """Update EMA parameters based on current model parameters."""
        if not self.apply_ema:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.ema_state, f"Parameter {name} not found in EMA state."
                ema_param = self.ema_state[name]
                # Update EMA: ema = decay * ema + (1 - decay) * param
                ema_param.mul_(self.ema_decay).add_(param, alpha=1.0 - self.ema_decay)

    def swap_parameters_with_ema(self, store_params_in_ema=True):
        """
        Swap model parameters with their EMA counterparts.
        If store_params_in_ema is True, the current parameters are stored in EMA before swapping.

        Args:
            store_params_in_ema (bool): Whether to store the current parameters in EMA.
        """
        if not self.apply_ema:
            warnings.warn('swap_parameters_with_ema was called when EMA is not enabled.')
            return

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            ema_param = self.ema_state[name]
            if store_params_in_ema:
                # Swap the current parameter with EMA
                temp = param.detach().clone()
                param.data.copy_(ema_param)
                self.ema_state[name].copy_(temp)
            else:
                # Replace parameter with EMA without updating EMA state
                param.data.copy_(ema_param)

    def state_dict(self):
        """Returns the state of the EMA."""
        return {name: param.cpu() for name, param in self.ema_state.items()}

    def load_state_dict(self, state_dict):
        """
        Loads the EMA state.

        Args:
            state_dict (dict): A dict containing EMA parameters.
        """
        for name, param in state_dict.items():
            if name in self.ema_state:
                self.ema_state[name].copy_(param.to(self.device))
            else:
                warnings.warn(f'EMA state_dict contains parameter "{name}" which is not in the model.')

    def to(self, device):
        """
        Moves EMA parameters to the specified device.

        Args:
            device (torch.device): The device to move EMA parameters to.
        """
        self.device = device
        for param in self.ema_state.values():
            param.data = param.data.to(device)
