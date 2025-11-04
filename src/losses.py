import torch
import torch.nn as nn

class SMAPELoss(nn.Module):
    def __init__(self, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
    def forward(self, pred, target):
        pred = pred.view_as(target)
        num = torch.abs(pred - target)
        den = (torch.abs(pred) + torch.abs(target)) / 2.0
        smape = num / (den + self.eps)
        if self.reduction == "mean":
            return smape.mean()
        elif self.reduction == "sum":
            return smape.sum()
        return smape

class CombinedLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.huber = nn.SmoothL1Loss(beta=0.8)
        self.smape = SMAPELoss(eps=eps)
    def forward(self, pred, target):
        return self.alpha * self.huber(pred, target) + (1.0 - self.alpha) * self.smape(pred, target)
