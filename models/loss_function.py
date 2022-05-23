import torch
from torch.autograd.gradcheck import gradcheck
import torch.nn.functional as F
import math
import sys
from torch.autograd import Variable
import numpy as np
import cv2

# The whole image rotation and translation loss cannot back propagation by cv2;
# But it can back propagation by affine_grid and grid_sample

######################
# Gradient Check True
# WholeImageRotationAndTranslation
# ShiftedLoss
# ImageBlockRotationAndTranslation
######################
# Gradient Check False
#
######################


def generate_theta(i_radian, i_tx, i_ty, i_batch_size, i_h, i_w, i_dtype):
    # if you want to keep ration when rotation a rectangle image
    # theta = torch.tensor([[math.cos(i_radian), math.sin(-i_radian) * i_h / i_w, i_tx],
    #                       [math.sin(i_radian) * i_w / i_h, math.cos(i_radian), i_ty]],
    #                      dtype=i_dtype).unsqueeze(0).repeat(i_batch_size, 1, 1)
    # else
    theta = torch.tensor([[math.cos(i_radian), math.sin(-i_radian), i_tx],
                          [math.sin(i_radian), math.cos(i_radian), i_ty]],
                         dtype=i_dtype).unsqueeze(0).repeat(i_batch_size, 1, 1)
    return theta


def rotate_mse_loss(i_fm1, i_fm2, i_mask):
    # the input feature map shape is (bs, 1, h, w)
    square_err = torch.mul(torch.pow((i_fm1 - i_fm2), 2), i_mask)
    mean_se = square_err.view(i_fm1.size(0), -1).sum(1) / i_mask.view(i_fm1.size(0), -1).sum(1)
    return mean_se


def mse_loss(src, target):
    if isinstance(src, torch.autograd.Variable):
        return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.data.nelement() * src.size(0)
    else:
        return ((src - target) ** 2).view(src.size(0), -1).sum(1) / src.nelement() * src.size(0)


class WholeImageRotationAndTranslation(torch.nn.Module):
    """
    The grid_sample is nondeterministic when using CUDA backend
    So I recommend to use CPU
    """
    def __init__(self, i_v_shift, i_h_shift, i_angle):
        super(WholeImageRotationAndTranslation, self).__init__()
        self.v_shift = i_v_shift
        self.h_shift = i_h_shift
        self.angle = i_angle

    def forward(self, i_fm1, i_fm2):
        b, c, h, w = i_fm1.shape
        mask = torch.ones_like(i_fm2, device=i_fm1.device)
        n_affine = 0
        if self.training:
            min_dist = torch.zeros([b, ], dtype=i_fm1.dtype, requires_grad=True, device=i_fm1.device)
        else:
            min_dist = torch.zeros([b, ], dtype=i_fm1.dtype, requires_grad=False, device=i_fm1.device)

        if self.v_shift == self.h_shift == 0:
            min_dist = mse_loss(i_fm1, i_fm2).cuda()
            return min_dist
        for tx in range(-self.h_shift, self.h_shift + 1):
            for ty in range(-self.v_shift, self.v_shift + 1):
                for a in range(-self.angle, self.angle + 1):
                    radian_a = a * math.pi / 180.
                    ratio_tx = 2 * tx / w
                    ratio_ty = 2 * ty / h
                    theta = generate_theta(radian_a, ratio_tx, ratio_ty, b, h, w, i_fm1.dtype)
                    grid = F.affine_grid(theta, i_fm2.size(), align_corners=False).to(i_fm1.device)
                    r_fm2 = F.grid_sample(i_fm2, grid, align_corners=False)
                    r_mask = F.grid_sample(mask, grid, align_corners=False)
                    # mean_se.shape: -> (bs, )
                    mean_se = rotate_mse_loss(i_fm1, r_fm2, r_mask)
                    if n_affine == 0:
                        min_dist = mean_se
                    else:
                        min_dist = torch.vstack([min_dist, mean_se])
                    n_affine += 1

        min_dist, _ = torch.min(min_dist, dim=0)
        return min_dist


class ImageBlockRotationAndTranslation(torch.nn.Module):
    def __init__(self, i_block_size, i_v_shift, i_h_shift, i_angle, i_topk=16):
        super(ImageBlockRotationAndTranslation, self).__init__()
        self.block_size = i_block_size
        self.v_shift = i_v_shift
        self.h_shift = i_h_shift
        self.angle = i_angle
        self.topk = i_topk

    def forward(self, i_fm1, i_fm2):
        b, c, h, w = i_fm1.shape
        if self.training:
            min_dist = torch.zeros([b, ], dtype=i_fm1.dtype, requires_grad=True, device=i_fm1.device)
        else:
            min_dist = torch.zeros([b, ], dtype=i_fm1.dtype, requires_grad=False, device=i_fm1.device)

        if self.v_shift == self.h_shift == self.angle == 0:
            min_dist = mse_loss(i_fm1, i_fm2).cuda()
            return min_dist

        n_affine = 0
        for sub_x in range(0, w, self.block_size):
            for sub_y in range(0, h, self.block_size):
                sub_fm1 = i_fm1[:, :, sub_y:sub_y + self.block_size, sub_x:sub_x + self.block_size]

                sub_affine = 0
                if self.training:
                    sub_min_dist = torch.zeros([b, ], dtype=i_fm1.dtype, requires_grad=True, device=i_fm1.device)
                else:
                    sub_min_dist = torch.zeros([b, ], dtype=i_fm1.dtype, requires_grad=False, device=i_fm1.device)

                for dx in range(-self.h_shift, self.h_shift + 1):
                    for dy in range(-self.v_shift, self.v_shift + 1):
                        if sub_y + dy < 0:
                            if sub_x + dx < 0:
                                sub_fm2 = i_fm2[:, :, 0:self.block_size, 0:self.block_size]
                            elif sub_x + dx + self.block_size > w:
                                sub_fm2 = i_fm2[:, :, 0:self.block_size, w - self.block_size:w]
                            else:
                                sub_fm2 = i_fm2[:, :, 0:self.block_size, sub_x + dx:sub_x + self.block_size + dx]
                        elif sub_y + dy + self.block_size > h:
                            if sub_x + dx < 0:
                                sub_fm2 = i_fm2[:, :, h - self.block_size:h, 0:self.block_size]
                            elif sub_x + dx + self.block_size > w:
                                sub_fm2 = i_fm2[:, :, h - self.block_size:h,
                                          w - self.block_size:w]
                            else:
                                sub_fm2 = i_fm2[:, :, h - self.block_size:h,
                                          sub_x + dx:sub_x + self.block_size + dx]
                        else:
                            if sub_x + dx < 0:
                                sub_fm2 = i_fm2[:, :, sub_y + dy:sub_y + self.block_size + dy, 0:self.block_size]
                            elif sub_x + dx + self.block_size > w:
                                sub_fm2 = i_fm2[:, :, sub_y + dy:sub_y + self.block_size + dy,
                                          w - self.block_size:w]
                            else:
                                sub_fm2 = i_fm2[:, :, sub_y + dy:sub_y + self.block_size + dy,
                                          sub_x + dx:sub_x + self.block_size + dx]

                        for a in range(-self.angle, self.angle + 1):
                            sub_fm2_b, sub_fm2_c, sub_fm2_h, sub_fm2_w = sub_fm2.shape
                            mask = torch.ones_like(sub_fm2, dtype=i_fm1.dtype, device=i_fm1.device)
                            radian_a = a * math.pi / 180.
                            theta = generate_theta(radian_a, 0, 0, sub_fm2_b, sub_fm2_h, sub_fm2_w, i_fm1.dtype)
                            grid = F.affine_grid(theta, sub_fm2.size(), align_corners=False.device)
                            r_sub_fm2 = F.grid_sample(sub_fm2, grid, align_corners=False)
                            r_mask = F.grid_sample(mask, grid, align_corners=False)
                            sub_mean_se = rotate_mse_loss(sub_fm1, r_sub_fm2, r_mask)
                            if sub_affine == 0:
                                sub_min_dist = sub_mean_se
                            else:
                                sub_min_dist = torch.vstack([sub_min_dist, sub_mean_se])
                            sub_affine += 1

                sub_min_dist, _ = torch.min(sub_min_dist, dim=0)
                if n_affine == 0:
                    min_dist = sub_min_dist
                else:
                    min_dist = torch.vstack([min_dist, sub_min_dist])
                n_affine += 1

        if self.training:
            min_dist = torch.sum(min_dist, dim=0)
        else:
            min_dist, _ = torch.topk(min_dist, self.topk, dim=0, largest=False)
            min_dist = torch.sum(min_dist, dim=0)

        return min_dist


class ShiftedLoss(torch.nn.Module):
    def __init__(self, hshift, vshift):
        super(ShiftedLoss, self).__init__()
        self.hshift = hshift
        self.vshift = vshift

    def forward(self, fm1, fm2):
        # C * H * W
        bs, _, h, w = fm1.size()
        if w < self.hshift:
            self.hshift = self.hshift % w
        if h < self.vshift:
            self.vshift = self.vshift % h

        # min_dist shape (bs, )  & sys.float_info.max is to get the max float value
        # min_dist save the maximal float value
        min_dist = torch.ones(bs).cuda() * sys.float_info.max
        # torch is set tensor as an instance of Variable
        if isinstance(fm1, torch.autograd.Variable):
            min_dist = Variable(min_dist, requires_grad=False)

        if self.hshift == 0 and self.vshift == 0:
            dist = mse_loss(fm1, fm2).cuda()
            min_dist, _ = torch.min(torch.stack([min_dist, dist]), 0)
            return min_dist

        for bh in range(-self.hshift, self.hshift + 1):
            for bv in range(-self.vshift, self.vshift + 1):
                if bh >= 0:
                    ref1, ref2 = fm1[:, :, :, :w - bh], fm2[:, :, :, bh:]
                else:
                    ref1, ref2 = fm1[:, :, :, -bh:], fm2[:, :, :, :w + bh]

                if bv >= 0:
                    ref1, ref2 = ref1[:, :, :h - bv, :], ref2[:, :, bv:, :]
                else:
                    ref1, ref2 = ref1[:, :, -bv:, :], ref2[:, :, :h + bv, :]
                dist = mse_loss(ref1, ref2).cuda()
                min_dist, _ = torch.min(torch.stack([min_dist.squeeze(), dist.squeeze()]), 0)
        return min_dist.squeeze()


class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, i_fm1, i_fm2):
        square_err = ((i_fm1 - i_fm2) ** 2).view(i_fm1.size(0), -1).sum(1)
        num = i_fm1.view(i_fm1.size(0), -1).size(1)
        mse = square_err / num
        return mse


class HammingDistance(torch.nn.Module):
    def __init__(self):
        super(HammingDistance, self).__init__()

    def forward(self, i_fm1, i_fm2):
        xor = torch.abs(torch.sub(i_fm1, i_fm2))
        hamming_distance = (xor.view(i_fm1.size(0), -1)).sum(1)
        return hamming_distance


if __name__ == "__main__":

    mode = "train"
    if mode == "eval":
        loss = WholeImageRotationAndTranslation(i_v_shift=3, i_h_shift=3, i_angle=3).eval()
        input1 = torch.randn([5, 5], dtype=torch.double, requires_grad=False).unsqueeze(0).unsqueeze(0)
        input2 = torch.randn([5, 5], dtype=torch.double, requires_grad=False).unsqueeze(0).unsqueeze(0)
        min_dist = loss(input1, input2)
        print(min_dist)
    else:
        loss = WholeImageRotationAndTranslation(i_v_shift=3, i_h_shift=3, i_angle=3).train()
        input1 = torch.randn([32, 32], dtype=torch.double, requires_grad=True).unsqueeze(0).unsqueeze(0)
        input2 = torch.randn([32, 32], dtype=torch.double, requires_grad=True).unsqueeze(0).unsqueeze(0)
        min_dist = loss(input1, input2)
        print(min_dist)
        test = gradcheck(loss, [input1, input2])
        print("Are the gradients of whole image rotation and translation with undeformable correct: ", test)


