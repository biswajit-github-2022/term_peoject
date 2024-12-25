
import torch


def calculate_miou(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        if y_pred.shape != y_true.shape:
        raise ValueError("Input tensors must have the same shape")

        y_pred = y_pred > 0.5
        intersection = torch.logical_and(y_pred, y_true)
    union = torch.logical_or(y_pred, y_true)

        iou_per_sample = torch.sum(intersection, dim=1) / torch.sum(union, dim=1)
        miou = torch.mean(iou_per_sample).item()

    return miou
