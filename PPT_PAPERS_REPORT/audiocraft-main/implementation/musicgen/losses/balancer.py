
import typing as tp

import flashy
import torch
from torch import autograd


class Balancer:
    def __init__(self, weights: tp.Dict[str, float], balance_grads: bool = True, total_norm: float = 1.,
                 ema_decay: float = 0.999, per_batch_item: bool = True, epsilon: float = 1e-12,
                 monitor: bool = False):
        self.weights = weights
        self.per_batch_item = per_batch_item
        self.total_norm = total_norm or 1.
        self.averager = flashy.averager(ema_decay or 1.)
        self.epsilon = epsilon
        self.monitor = monitor
        self.balance_grads = balance_grads
        self._metrics: tp.Dict[str, tp.Any] = {}

    @property
    def metrics(self):
        return self._metrics

    def backward(self, losses: tp.Dict[str, torch.Tensor], input: torch.Tensor) -> torch.Tensor:
        norms = {}
        grads = {}
        for name, loss in losses.items():
                        grad, = autograd.grad(loss, [input], retain_graph=True)
            if self.per_batch_item:
                                dims = tuple(range(1, grad.dim()))
                norm = grad.norm(dim=dims, p=2).mean()
            else:
                norm = grad.norm(p=2)
            norms[name] = norm
            grads[name] = grad

        count = 1
        if self.per_batch_item:
            count = len(grad)
                        avg_norms = flashy.distrib.average_metrics(self.averager(norms), count)
                        total = sum(avg_norms.values())

        self._metrics = {}
        if self.monitor:
                        for k, v in avg_norms.items():
                self._metrics[f'ratio_{k}'] = v / total

        total_weights = sum([self.weights[k] for k in avg_norms])
        assert total_weights > 0.
        desired_ratios = {k: w / total_weights for k, w in self.weights.items()}

        out_grad = torch.zeros_like(input)
        effective_loss = torch.tensor(0., device=input.device, dtype=input.dtype)
        for name, avg_norm in avg_norms.items():
            if self.balance_grads:
                                scale = desired_ratios[name] * self.total_norm / (self.epsilon + avg_norm)
            else:
                                scale = self.weights[name]
            out_grad.add_(grads[name], alpha=scale)
            effective_loss += scale * losses[name].detach()
                input.backward(out_grad)
        return effective_loss
