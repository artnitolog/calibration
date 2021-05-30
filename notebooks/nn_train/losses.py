import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        assert 0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.conf = 1.0 - smoothing
        self.reduction = reduction
        
    def _reduction(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
    
    def forward(self, input, target):
        '''
        Args:
            input: tensor (n, n_classes), logits
            target: tensor (n,) of true labels in range(0, n_classes)
        '''
        log_probs = F.log_softmax(input, dim=1)
        nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze()
        smoothed = -log_probs.mean(dim=1)
        loss = nll * self.conf + smoothed * self.smoothing
        return self._reduction(loss)


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
    def _reduction(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
    
    def forward(self, input, target):
        '''
        Args:
            input: tensor (n, n_classes), logits
            target: tensor (n,) of true labels in range(0, n_classes)
        '''
        log_probs = F.log_softmax(input, dim=1)
        log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze()
        probs = log_probs.exp()
        loss = -torch.pow(1 - probs, self.gamma) * log_probs
        return self._reduction(loss)
