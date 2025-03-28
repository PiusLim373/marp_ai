import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN


class MaskedFCNet(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_dim = 34
        self.mask_dim = 25
        hidden_dim = 128

        # Simple 4-layer fully connected network
        self.policy_net = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_outputs),
        )

        self._value_branch = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._value = None

    def forward(self, input_dict, state, seq_lens):
        if isinstance(input_dict["obs"], dict):
            obs_tensor = input_dict["obs"]["obs"]
            action_mask = input_dict["obs"]["action_mask"]
        else:
            obs_tensor = input_dict["obs"][:, :34]
            action_mask = input_dict["obs"][:, 34:]

        logits = self.policy_net(obs_tensor)
        self._value = self._value_branch(obs_tensor).squeeze(1)

        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        return masked_logits, state


    def value_function(self):
        return self._value
