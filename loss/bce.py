import torch
import torch.nn as nn
import torch.nn.functional as F


# Binary Segmentation loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, avg=True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])  # 对背景乘alpha，对前景乘(1-alpha)
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)  # 可以自己指定alpha
        self.avg = avg

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.contiguous().view(input.size(0), input.size(1), -1)

            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)

            # N,H*W,C => N*H*W,C
            input = input.contiguous().view(-1, input.size(2))

        # (N,H,W) => (N*H*W,1)
        target = target.contiguous().view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target.long())  # (N*H*W,2) and (N*H*W,1)
