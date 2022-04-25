import torch
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F


class ImageBlockRotationAndTranslationLoss(torch.nn.Module):
    def __int__(self, i_block_size, i_v_shift, i_h_shift, i_top_k):
        super(ImageBlockRotationAndTranslationLoss, self).__int__()
        self.block_size = i_block_size
        self.v_shift = i_v_shift
        self.h_shift = i_h_shift
        self.top_k = i_top_k

    def forward(self, i_fm1: torch.Tensor, i_fm2: torch.Tensor):
        bs, _, h, w = i_fm1.shape
        loss = torch.zeros((bs,), requires_grad=True)

        return loss