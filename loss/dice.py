import torch.nn as nn
import torch
import torch.nn.functional as F

# Segmentation
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps =1e-7

    def forward(self, logits, true):
        """

        :param logits: (N,C,H,W)   分别表示batch, channel, height, width
        :param true: (N,H,W)
        :return: loss
        """
        num_classes = logits.shape[1]
        logits = logits.to(torch.device('cpu'))
        true = true.to(torch.device('cpu'))
        if num_classes == 1:  # num_classes==2
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1).int()] # eye 这个函数主要是为了生成对角线全1，其余部分全0的二维数组
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()  # [B,H,W,2] => [B,2,H,W]
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension() + 1))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return 1 - dice_loss