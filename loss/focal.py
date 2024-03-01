import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Binary Segmentation loss
class FocalLoss(nn.Module):
    """
    when gamma=0, the Focal loss function becomes the BalanCE(BCE) loss.

    :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
    :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
    """
    def __init__(self, gamma=0, alpha=None, avg=True):
        super().__init__()
        self.gamma = gamma  # gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])  # 对背景乘alpha，对前景乘(1-alpha)
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)  # 可以自己指定alpha
        self.avg = avg

    def forward(self, input, target):
        """

        :param input: (N,C,H,W)   分别表示batch, channel, height, width
        :param target: (N,H,W)
        :return: loss
        """
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
        logpt = logpt.gather(1, target.long())  # (N*H*W,2) and (N*H*W,1) ---> (N*H*W,1)
        logpt = logpt.view(-1)  # (N*H*W,1) => (N*H*W)  (-inf,0]
        pt = Variable(logpt.data.exp())  # [0,1]

        if self.alpha is not None:
            '''
            alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
            '''
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt  # gamma==0

        # fore = loss[target.view(-1)==1].data
        # back = loss[target.view(-1)==0].data
        # print("fore: {} back: {} ".format(fore.sum(), back.sum()))   # 打印前景与背景的loss值

        if self.avg:
            return loss.mean()
        else:
            return loss.sum()
