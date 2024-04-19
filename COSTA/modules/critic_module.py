import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Optional


class Critic(nn.Module):
    def __init__(self, backbone: nn.Module, device: str = "cpu", positive: bool = False, max_value = 0.0) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        self.last = nn.Linear(latent_dim, 1).to(device)
        self.positive = positive
        self.soft_plus = nn.ReLU()
        self.max_value = max_value

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32).flatten(1)
            obs = torch.cat([obs, actions], dim=1)
        logits = self.backbone(obs)
        values = self.last(logits)
        if self.positive:
            values = self.soft_plus(values)
        if self.max_value > 0:
            if values.std() > self.max_value:
                norm_values = (values - values.mean())/values.std()
                norm_values = norm_values - norm_values.min()
                norm_values = norm_values * self.max_value
                return values, norm_values
            return values, values
        return values