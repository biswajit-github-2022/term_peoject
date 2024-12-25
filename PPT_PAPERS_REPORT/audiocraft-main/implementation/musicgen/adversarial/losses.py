

import typing as tp

import flashy
import torch
import torch.nn as nn
import torch.nn.functional as F


ADVERSARIAL_LOSSES = ['mse', 'hinge', 'hinge2']


AdvLossType = tp.Union[nn.Module, tp.Callable[[torch.Tensor], torch.Tensor]]
FeatLossType = tp.Union[nn.Module, tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]


class AdversarialLoss(nn.Module):
    def __init__(self,
                 adversary: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss: AdvLossType,
                 loss_real: AdvLossType,
                 loss_fake: AdvLossType,
                 loss_feat: tp.Optional[FeatLossType] = None,
                 normalize: bool = True):
        super().__init__()
        self.adversary: nn.Module = adversary
        flashy.distrib.broadcast_model(self.adversary)
        self.optimizer = optimizer
        self.loss = loss
        self.loss_real = loss_real
        self.loss_fake = loss_fake
        self.loss_feat = loss_feat
        self.normalize = normalize

    def _save_to_state_dict(self, destination, prefix, keep_vars):
                super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'optimizer'] = self.optimizer.state_dict()
        return destination

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
                self.optimizer.load_state_dict(state_dict.pop(prefix + 'optimizer'))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def get_adversary_pred(self, x):

        We assume the adversary output is the following format: Tuple[List[torch.Tensor], List[List[torch.Tensor]]].
        The first item being the logits and second item being a list of feature maps for each sub-discriminator.

        This will automatically synchronize gradients (with `flashy.distrib.eager_sync_model`)
        and call the optimizer.
        and feature matching loss if provided.

    Args:
        loss (nn.Module): Loss to use for feature matching (default=torch.nn.L1).
        normalize (bool): Whether to normalize the loss.
            by number of feature maps.
