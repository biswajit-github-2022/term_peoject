
import typing as tp
import random

import torch


def pad(
    x_wm: torch.Tensor, central: bool = False
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        max_start = int(0.33 * x_wm.size(-1))
    min_end = int(0.66 * x_wm.size(-1))
    starts = torch.randint(0, max_start, size=(x_wm.size(0),))
    ends = torch.randint(min_end, x_wm.size(-1), size=(x_wm.size(0),))
    mask = torch.zeros_like(x_wm)
    for i in range(x_wm.size(0)):
        mask[i, :, starts[i]: ends[i]] = 1
    if central:
        mask = 1 - mask
    padded = x_wm * mask
    true_predictions = torch.cat([1 - mask, mask], dim=1)
    return padded, true_predictions


def mix(
    x: torch.Tensor, x_wm: torch.Tensor, window_size: float = 0.5, shuffle: bool = False
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    assert 0 < window_size <= 1, "window_size should be between 0 and 1"

        max_start_point = x.shape[-1] - int(window_size * x.shape[-1])

        start_point = random.randint(0, max_start_point)

        total_frames = x.shape[-1]
    window_frames = int(window_size * total_frames)

                mixed = x_wm.detach().clone()

    true_predictions = torch.cat(
        [torch.zeros_like(mixed), torch.ones_like(mixed)], dim=1
    )
        true_predictions[:, 0, start_point: start_point + window_frames] = 1.0
        true_predictions[:, 1, start_point: start_point + window_frames] = 0.0

    if shuffle:
                shuffle_idx = torch.randint(0, x.size(0), (x.size(0),))
        mixed[:, :, start_point: start_point + window_frames] = x[shuffle_idx][
            :, :, start_point: start_point + window_frames
        ]
    else:
        mixed[:, :, start_point: start_point + window_frames] = x[
            :, :, start_point: start_point + window_frames
        ]

    return mixed, true_predictions
