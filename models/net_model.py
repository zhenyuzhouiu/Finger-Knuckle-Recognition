# =========================================================
# @ Network Architecture File: Residual Feature Network 128
# @ Target dataset: IITD, PolyU 2D/3D Contactless, CASIA,
#   300 Subject, 600 Subject, 35 Subject
# =========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.net_common import ConvLayer, ResidualBlock, \
    DeformableConv2d2v, LKA, DeConvResBlock, PatchEmbed, Block
from models.efficientnet import EfficientVSResidual


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.zeros_(m.bias)
        torch.nn.init.ones_(m.weight)


class ResidualFeatureNet(torch.nn.Module):
    def __init__(self):
        super(ResidualFeatureNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.resid1 = ResidualBlock(128)
        self.resid2 = ResidualBlock(128)
        self.resid3 = ResidualBlock(128)
        self.resid4 = ResidualBlock(128)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.conv5 = ConvLayer(64, 1, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        resid1 = self.resid1(conv3)
        resid2 = self.resid1(resid1)
        resid3 = self.resid1(resid2)
        resid4 = self.resid1(resid3)
        conv4 = F.relu(self.conv4(resid4))
        conv5 = F.relu(self.conv5(conv4))

        return conv5


class RFNBinaryConvNet(torch.nn.Module):
    def __init__(self):
        super(RFNBinaryConvNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.resid1 = ResidualBlock(128)
        self.resid2 = ResidualBlock(128)
        self.resid3 = ResidualBlock(128)
        self.resid4 = ResidualBlock(128)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.conv5 = ConvLayer(64, 1, kernel_size=1, stride=1)
        self.bn5 = nn.BatchNorm2d(num_features=1)

        self.threshold = nn.Sequential(
            nn.Conv2d(128, 128, groups=128, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, groups=128, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, groups=128, kernel_size=4),
            nn.Sigmoid()
        )

    def binary(self, x):
        # x.shape:-> [bs, 128, 32, 32]
        # xs.shape:-> [bs, 128, 1, 1]
        xs = self.threshold(x)
        xs = torch.sign(x - xs)
        x = torch.relu(xs)
        return x

    def forward(self, x):
        conv1 = F.relu(self.bn1(self.conv1(x)))
        conv2 = F.relu(self.bn2(self.conv2(conv1)))
        conv3 = F.relu(self.bn3(self.conv3(conv2)))
        resid1 = self.resid1(conv3)
        resid2 = self.resid1(resid1)
        resid3 = self.resid1(resid2)
        resid4 = self.resid1(resid3)
        binary = self.binary(resid4)
        conv4 = F.relu(torch.sign(self.conv4(binary)))
        conv5 = F.relu(torch.sign(self.conv5(conv4)))

        return conv5


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.bn3 = torch.nn.BatchNorm2d(num_features=128)
        self.conv4 = ConvLayer(128, 64, kernel_size=1, stride=1)
        self.bn4 = torch.nn.BatchNorm2d(num_features=64)
        self.conv5 = ConvLayer(64, 1, kernel_size=1, stride=1)
        self.bn5 = torch.nn.BatchNorm2d(num_features=1)

    def forward(self, x):
        conv1 = F.relu(self.bn1(self.conv1(x)))
        conv2 = F.relu(self.bn2(self.conv2(conv1)))
        conv3 = F.relu(self.bn3(self.conv3(conv2)))
        conv4 = F.relu(self.bn4(self.conv4(conv3)))
        conv5 = F.relu(self.bn5(self.conv5(conv4)))

        return conv5


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(5, 5, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(5 * 28 * 28, 5 * 28),
            nn.ReLU(True),
            nn.Linear(5 * 28, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 5 * 28 * 28)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), )
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        return x


class STNetConvNet(torch.nn.Module):
    def __init__(self):
        super(STNetConvNet, self).__init__()
        # Spatial Transformer Network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7, bias=False),
            nn.BatchNorm2d(num_features=8),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5, bias=False),
            nn.BatchNorm2d(num_features=10),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 28 * 28, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.bn3 = torch.nn.BatchNorm2d(num_features=128)
        self.conv4 = ConvLayer(128, 64, kernel_size=1, stride=1)
        self.bn4 = torch.nn.BatchNorm2d(num_features=64)
        self.conv5 = ConvLayer(64, 1, kernel_size=1, stride=1)
        self.bn5 = torch.nn.BatchNorm2d(num_features=1)

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 28 * 28)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    def forward(self, x):
        x = self.stn(x)
        conv1 = F.relu(self.bn1(self.conv1(x)))
        conv2 = F.relu(self.bn2(self.conv2(conv1)))
        conv3 = F.relu(self.bn3(self.conv3(conv2)))
        conv4 = F.relu(self.bn4(self.conv4(conv3)))
        conv5 = F.relu(self.bn5(self.conv5(conv4)))

        return conv5


class ConvNetEfficientNet(torch.nn.Module):
    def __init__(self):
        super(ConvNetEfficientNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.efficient = EfficientVSResidual(width_coefficient=1, depth_coefficient=1, drop_connect_rate=0.)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.conv5 = ConvLayer(64, 1, kernel_size=1, stride=1)

    def forward(self, x):
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        efficient = self.efficient(conv3)
        conv4 = F.relu(self.conv4(efficient))
        conv5 = F.relu(self.conv5(conv4))

        return conv5


class STNetConvNetEfficientNet(torch.nn.Module):
    def __init__(self):
        super(STNetConvNetEfficientNet, self).__init__()
        # Spatial Transformer Network
        self.localization = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=7, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=5, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        # Initial convolution layers
        # input shape:-> [bs, c, h ,w]:[bs, 3, 128, 128]
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(num_features=128)
        # output shape:-> [bs, 128, 32, 32]
        self.efficient = EfficientVSResidual(width_coefficient=1, depth_coefficient=1, drop_connect_rate=0.)
        self.conv4 = ConvLayer(128, 64, kernel_size=1, stride=1, bias=False)
        self.bn4 = torch.nn.BatchNorm2d(num_features=64)
        self.conv5 = ConvLayer(64, 1, kernel_size=1, stride=1)
        # self.bn5 = torch.nn.BatchNorm2d(num_features=1)

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    def forward(self, x):
        conv1 = F.relu(self.bn1(self.conv1(x)))
        conv2 = F.relu(self.bn2(self.conv2(conv1)))
        conv3 = F.relu(self.bn3(self.conv3(conv2)))
        stn_conv3 = self.stn(conv3)
        efficient = self.efficient(stn_conv3)
        conv4 = F.relu(self.bn4(self.conv4(efficient)))
        conv5 = F.relu(self.conv5(conv4))

        return conv5


class FirstSTNetThenConvNetMiddleEfficientNet(torch.nn.Module):
    def __init__(self):
        super(FirstSTNetThenConvNetMiddleEfficientNet, self).__init__()
        # Spatial Transformer Network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7, bias=False),
            nn.BatchNorm2d(num_features=8),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=5, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(16 * 28 * 28, 28 * 28),
            nn.ReLU(True),
            nn.Linear(28 * 28, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[4].weight.data.zero_()
        self.fc_loc[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        # Initial convolution layers
        # input shape:-> [bs, c, h ,w]:[bs, 3, 128, 128]
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(num_features=128)
        # output shape:-> [bs, 128, 32, 32]
        self.efficient = EfficientVSResidual(width_coefficient=1, depth_coefficient=1, drop_connect_rate=0.)
        self.conv4 = ConvLayer(128, 64, kernel_size=1, stride=1, bias=False)
        self.bn4 = torch.nn.BatchNorm2d(num_features=64)
        self.conv5 = ConvLayer(64, 1, kernel_size=1, stride=1)
        # self.bn5 = torch.nn.BatchNorm2d(num_features=1)

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 16 * 28 * 28)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    def forward(self, x):
        stn_x = self.stn(x)
        conv1 = F.relu(self.bn1(self.conv1(stn_x)))
        conv2 = F.relu(self.bn2(self.conv2(conv1)))
        conv3 = F.relu(self.bn3(self.conv3(conv2)))
        efficient = self.efficient(conv3)
        conv4 = F.relu(self.bn4(self.conv4(efficient)))
        conv5 = F.relu(self.conv5(conv4))

        return conv5


class ImageBlocksRFNet(torch.nn.Module):
    def __init__(self):
        super(ImageBlocksRFNet, self).__init__()
        self.conv1 = ConvLayer(48, 512, kernel_size=5, stride=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(num_features=512)
        self.conv2 = ConvLayer(512, 1024, kernel_size=3, stride=2, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(num_features=1024)
        self.conv3 = ConvLayer(1024, 2048, kernel_size=3, stride=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(num_features=2048)
        self.resid1 = ResidualBlock(2048)
        self.resid2 = ResidualBlock(2048)
        self.resid3 = ResidualBlock(2048)
        self.resid4 = ResidualBlock(2048)
        self.conv4 = ConvLayer(2048, 1024, kernel_size=3, stride=1, bias=False)
        self.bn4 = torch.nn.BatchNorm2d(num_features=1024)
        self.conv5 = ConvLayer(1024, 512, kernel_size=3, stride=1, bias=False)
        self.bn5 = torch.nn.BatchNorm2d(num_features=512)
        self.conv6 = ConvLayer(512, 16, kernel_size=1, stride=1, bias=False)
        self.bn6 = torch.nn.BatchNorm2d(num_features=16)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x.shape:-> [bs,3,128,128]
        bs, c, h, w = x.shape
        x = torch.cat((x[:, :, :32, :32], x[:, :, 32:64, :32], x[:, :, 64:96, :32], x[:, :, 96:128, :32],
                       x[:, :, :32, 32:64], x[:, :, 32:64, 32:64], x[:, :, 64:96, 32:64], x[:, :, 96:128, 32:64],
                       x[:, :, :32, 64:96], x[:, :, 32:64, 64:96], x[:, :, 64:96, 64:96], x[:, :, 96:128, 64:96],
                       x[:, :, :32, 96:], x[:, :, 32:64, 96:], x[:, :, 64:96, 96:], x[:, :, 96:128, 96:]),
                      dim=1)
        conv1 = F.relu(self.bn1(self.conv1(x)))
        conv2 = F.relu(self.bn2(self.conv2(conv1)))
        conv3 = F.relu(self.bn3(self.conv3(conv2)))
        resid1 = self.resid1(conv3)
        resid2 = self.resid1(resid1)
        resid3 = self.resid1(resid2)
        resid4 = self.resid1(resid3)
        conv4 = F.relu(self.bn4(self.conv4(resid4)))
        # conv5.shape:-> [bs, 16, 8, 8]
        conv5 = F.relu(self.bn5(self.conv5(conv4)))
        conv6 = F.relu(self.bn6(self.conv6(conv5)))
        out = torch.zeros([bs, 1, 32, 32], device=x.device)
        out[:, :, :8, :8] = conv6[:, 0, :, :].unsqueeze(1)
        out[:, :, 8:16, :8] = conv6[:, 1, :, :].unsqueeze(1)
        out[:, :, 16:24, :8] = conv6[:, 2, :, :].unsqueeze(1)
        out[:, :, 24:, :8] = conv6[:, 3, :, :].unsqueeze(1)
        out[:, :, :8, 8:16] = conv6[:, 4, :, :].unsqueeze(1)
        out[:, :, 8:16, 8:16] = conv6[:, 5, :, :].unsqueeze(1)
        out[:, :, 16:24, 8:16] = conv6[:, 6, :, :].unsqueeze(1)
        out[:, :, 24:, 8:16] = conv6[:, 7, :, :].unsqueeze(1)
        out[:, :, :8, 16:24] = conv6[:, 8, :, :].unsqueeze(1)
        out[:, :, 8:16, 16:24] = conv6[:, 9, :, :].unsqueeze(1)
        out[:, :, 16:24, 16:24] = conv6[:, 10, :, :].unsqueeze(1)
        out[:, :, 24:, 16:24] = conv6[:, 11, :, :].unsqueeze(1)
        out[:, :, :8, 24:] = conv6[:, 12, :, :].unsqueeze(1)
        out[:, :, 8:16, 24:] = conv6[:, 13, :, :].unsqueeze(1)
        out[:, :, 16:24, 24:] = conv6[:, 14, :, :].unsqueeze(1)
        out[:, :, 24:, 24:] = conv6[:, 15, :, :].unsqueeze(1)
        return out


class RFNWithSTNet(torch.nn.Module):
    def __init__(self):
        super(RFNWithSTNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)
        self.bn3 = torch.nn.BatchNorm2d(num_features=128)
        self.resid1 = ResidualBlock(128)
        self.resid2 = ResidualBlock(128)
        self.resid3 = ResidualBlock(128)
        self.resid4 = ResidualBlock(128)
        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.bn4 = torch.nn.BatchNorm2d(num_features=64)
        self.conv5 = ConvLayer(64, 1, kernel_size=1, stride=1)
        self.bn5 = torch.nn.BatchNorm2d(num_features=1)
        # output shape: [bs, 1, 32, 32]

        # Spatial Transformer Network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    def forward(self, x):
        conv1 = F.relu(self.bn1(self.conv1(x)))
        conv2 = F.relu(self.bn2(self.conv2(conv1)))
        conv3 = F.relu(self.bn3(self.conv3(conv2)))
        resid1 = self.resid1(conv3)
        resid2 = self.resid1(resid1)
        resid3 = self.resid1(resid2)
        resid4 = self.resid1(resid3)
        conv4 = F.relu(self.bn4(self.conv4(resid4)))
        conv5 = F.relu(self.bn5(self.conv5(conv4)))

        out = self.stn(conv5)

        return out


class DeConvRFNet(torch.nn.Module):
    def __init__(self):
        super(DeConvRFNet, self).__init__()
        # Initial convolution layers
        self.conv1 = DeformableConv2d2v(3, 32, kernel_size=5, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.conv2 = DeformableConv2d2v(32, 64, kernel_size=3, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        self.conv3 = DeformableConv2d2v(64, 128, kernel_size=3, stride=1)
        self.bn3 = torch.nn.BatchNorm2d(num_features=128)
        self.resid1 = DeConvResBlock(128)
        self.resid2 = DeConvResBlock(128)
        self.resid3 = DeConvResBlock(128)
        self.resid4 = DeConvResBlock(128)
        self.conv4 = DeformableConv2d2v(128, 64, kernel_size=3, stride=1)
        self.bn4 = torch.nn.BatchNorm2d(num_features=64)
        self.conv5 = ConvLayer(64, 1, kernel_size=1, stride=1)
        self.bn5 = torch.nn.BatchNorm2d(num_features=1)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        conv1 = F.relu(self.bn1(self.conv1(x)))
        conv2 = F.relu(self.bn2(self.conv2(conv1)))
        conv3 = F.relu(self.bn3(self.conv3(conv2)))
        resid1 = self.resid1(conv3)
        resid2 = self.resid1(resid1)
        resid3 = self.resid1(resid2)
        resid4 = self.resid1(resid3)
        conv4 = F.relu(self.bn4(self.conv4(resid4)))
        # conv5 = F.relu(self.bn5(self.conv5(conv4)))
        conv5 = F.relu(self.conv5(conv4))

        return conv5


class TNet_16(torch.nn.Module):
    """
       depth: the number of attention layer
       representation_size: the number of final MLP Head's Pre-Logits node,
                            if the value is none, then the head will not construct the Pre-Logits layer
       """

    def __init__(self, img_size=128, patch_size=16, in_c=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(TNet_16, self).__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(torch.nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or torch.nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # trainable position embedding
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = torch.nn.Dropout(p=drop_ratio)

        # drop_path_ratio is the number layer of Encoder Block
        # torch.linspace(start, end, steps)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = torch.nn.Sequential(
            *[Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                    norm_layer=norm_layer, act_layer=act_layer)
              for i in range(depth)
              ])
        self.norm = norm_layer(embed_dim)
        self.conv = torch.nn.Conv2d(embed_dim, 1, 1, 1)
        # Weight init
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 64, 768]

        # directly element add
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, C, 8, 8)
        x = self.conv(x)
        x = F.relu(x)

        return x


class TNet_8(torch.nn.Module):
    """
         depth: the number of attention layer
         representation_size: the number of final MLP Head's Pre-Logits node,
                              if the value is none, then the head will not construct the Pre-Logits layer
         """

    def __init__(self, img_size=128, patch_size=8, in_c=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(TNet_8, self).__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(torch.nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or torch.nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # trainable position embedding
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = torch.nn.Dropout(p=drop_ratio)

        # drop_path_ratio is the number layer of Encoder Block
        # torch.linspace(start, end, steps)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = torch.nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.conv = torch.nn.Conv2d(embed_dim, 1, 1, 1)
        # Weight init
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 256, 768]

        # directly element add
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, C, 16, 16)
        x = self.conv(x)
        x = F.relu(x)

        return x


class CTNet(torch.nn.Module):
    def __init__(self, norm_layer=None, act_layer=None):
        super(CTNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)

        norm_layer = norm_layer or partial(torch.nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or torch.nn.GELU
        self.patch_embed1 = PatchEmbed(img_size=64, patch_size=8, in_c=128, embed_dim=64)
        num_patches1 = self.patch_embed1.num_patches
        self.pos_embed1 = torch.nn.Parameter(torch.zeros(1, num_patches1, 64))
        self.pos_drop1 = torch.nn.Dropout(p=0.)
        self.blocks1 = torch.nn.Sequential(*[
            Block(dim=64, num_heads=2, mlp_ratio=4, qkv_bias=True,
                  qk_scale=None,
                  drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(1)
        ])
        self.norm1 = norm_layer(64)
        self.conv3 = torch.nn.Conv2d(64, 1, 1, 1)

        torch.nn.init.trunc_normal_(self.pos_embed1, std=0.02)

        self.apply(_init_vit_weights)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.patch_embed1(x)
        x = self.pos_drop1(x + self.pos_embed1)
        x = self.blocks1(x)
        x = self.norm1(x)

        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, C, 8, 8)
        x = self.conv3(x)
        return x


class OldCTNet(torch.nn.Module):
    def __init__(self, norm_layer=None, act_layer=None):
        super(CTNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.maxpooling = torch.nn.MaxPool2d(2, 2)

        norm_layer = norm_layer or partial(torch.nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or torch.nn.GELU
        self.patch_embed1 = PatchEmbed(img_size=64, patch_size=2, in_c=64, embed_dim=256)
        num_patches1 = self.patch_embed1.num_patches
        self.pos_embed1 = torch.nn.Parameter(torch.zeros(1, num_patches1, 256))
        self.pos_drop1 = torch.nn.Dropout(p=0.)
        self.blocks1 = torch.nn.Sequential(*[
            Block(dim=256, num_heads=2, mlp_ratio=4, qkv_bias=True,
                  qk_scale=None,
                  drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(1)
        ])
        self.norm1 = norm_layer(256)

        self.conv = torch.nn.Conv2d(256, 1, 1, 1)

        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)

        self.patch_embed2 = PatchEmbed(img_size=16, patch_size=2, in_c=128, embed_dim=128)
        num_patches2 = self.patch_embed2.num_patches
        self.pos_embed2 = torch.nn.Parameter(torch.zeros(1, num_patches2, 128))
        self.pos_drop2 = torch.nn.Dropout(p=0.)
        self.blocks2 = torch.nn.Sequential(*[
            Block(dim=128, num_heads=1, mlp_ratio=4, qkv_bias=True,
                  qk_scale=None,
                  drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(1)
        ])
        self.norm2 = norm_layer(128)

        self.conv4 = torch.nn.Conv2d(128, 1, 1, 1)

        torch.nn.init.trunc_normal_(self.pos_embed1, std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embed2, std=0.02)

        self.apply(_init_vit_weights)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.patch_embed1(x)
        x = self.pos_drop1(x + self.pos_embed1)
        x = self.blocks1(x)
        x = self.norm1(x)

        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, C, 32, 32)
        y = F.relu(self.conv(x))

        x = F.relu(self.conv3(x))

        x = self.patch_embed2(x)
        x = self.pos_drop2(x + self.pos_embed2)
        x = self.blocks2(x)
        x = self.norm2(x)

        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, C, 8, 8)

        # x = F.relu(self.conv4(x))
        return y


class DeConvLAKNet(torch.nn.Module):
    def __init__(self):
        super(DeConvLAKNet, self).__init__()
        self.conv1 = DeformableConv2d2v(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = DeformableConv2d2v(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.lka1 = LKA(dim=128)
        self.conv3 = DeformableConv2d2v(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = DeformableConv2d2v(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.lka2 = LKA(dim=128)
        self.conv5 = DeformableConv2d2v(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = DeformableConv2d2v(in_channels=64, out_channels=1, kernel_size=1, stride=1)

        self.apply(_init_vit_weights)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.lka1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.lka2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        return x


class ConvLAKNet(torch.nn.Module):
    def __init__(self):
        super(ConvLAKNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.lka1 = LKA(dim=128)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.lka2 = LKA(dim=128)
        self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1)

        self.apply(_init_vit_weights)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.lka1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.lka2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        return x


class DeepCLAKNet(torch.nn.Module):
    def __init__(self):
        super(DeepCLAKNet, self).__init__()
        # output feature map 32x128x128
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.lak1 = LKA(dim=32)

        # output feature map 128x64x64
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.bn3 = torch.nn.BatchNorm2d(num_features=64)
        self.lak2 = LKA(dim=64)

        # output feature map 256x256
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(num_features=128)
        self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.bn5 = torch.nn.BatchNorm2d(num_features=128)
        self.lak3 = LKA(dim=128)

        # add all feature map
        # 128-64
        self.conv6 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.bn6 = torch.nn.BatchNorm2d(num_features=64)
        # 64-32
        self.conv7 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.bn7 = torch.nn.BatchNorm2d(num_features=128)

        # output feature 1x32x32
        self.conv8 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv9 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x128 = self.lak1(x)

        x = F.relu(self.bn2(self.conv2(x128)))
        x = F.relu(self.bn3(self.conv3(x)))
        x64 = self.lak2(x)

        x = F.relu(self.bn4(self.conv4(x64)))
        x = F.relu(self.bn5(self.conv5(x)))
        x32 = self.lak3(x)

        x64 = x64 + F.relu(self.bn6(self.conv6(x128)))
        x32 = x32 + F.relu(self.bn7(self.conv7(x64)))

        out = F.relu(self.conv8(x32))
        out = F.relu(self.conv9(out))

        return out


class MultiCLAKNet(torch.nn.Module):
    def __init__(self):
        super(MultiCLAKNet, self).__init__()
        # output feature map 32x128x128
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.lak1 = LKA(dim=32)

        # output feature map 64x64x64
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.bn3 = torch.nn.BatchNorm2d(num_features=64)
        self.lak2 = LKA(dim=64)

        # output feature map 128x32x32
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(num_features=128)
        self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.bn5 = torch.nn.BatchNorm2d(num_features=128)
        self.lak3 = LKA(dim=128)

        # output feature map 128x16x16
        self.conv6 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn6 = torch.nn.BatchNorm2d(num_features=128)
        self.conv7 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.bn7 = torch.nn.BatchNorm2d(num_features=128)

        # concat all feature map
        # 128-64
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # 64-32
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # 16-32
        self.conv10 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3,
                                               stride=2, padding=1, dilation=1, output_padding=1)
        self.bn10 = torch.nn.BatchNorm2d(num_features=128)

        # input output feature 1x32x32
        self.conv11 = torch.nn.Conv2d(in_channels=352, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn11 = torch.nn.BatchNorm2d(num_features=256)
        self.conv12 = torch.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn12 = torch.nn.BatchNorm2d(num_features=64)
        self.conv13 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x128 = self.lak1(x)

        x = F.relu(self.bn2(self.conv2(x128)))
        x = F.relu(self.bn3(self.conv3(x)))
        x64 = self.lak2(x)

        x = F.relu(self.bn4(self.conv4(x64)))
        x = F.relu(self.bn5(self.conv5(x)))
        x32 = self.lak3(x)

        x = F.relu(self.bn6(self.conv6(x32)))
        x16 = F.relu(self.bn7(self.conv7(x)))

        x64 = torch.cat([x64, self.pool1(x128)], dim=1)
        x32 = torch.cat([x32, self.pool2(x64)], dim=1)
        x32 = torch.cat([x32, F.relu(self.bn10(self.conv10(x16)))], dim=1)

        out = F.relu(self.bn11(self.conv11(x32)))
        out = F.relu(self.bn12(self.conv12(out)))
        out = F.relu(self.conv13(out))

        return out


class ConvNetEfficientSTNetBinary(nn.Module):
    def __init__(self):
        super(ConvNetEfficientSTNetBinary, self).__init__()
        # Initial convolution layers
        self.feature = nn.Sequential(
            # input shape:-> [bs, 3, 128, 128]
            ConvLayer(3, 32, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            ConvLayer(32, 64, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            ConvLayer(64, 128, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            # EfficientNet MBlock
            EfficientVSResidual(width_coefficient=1, depth_coefficient=1, drop_connect_rate=0.),
            ConvLayer(128, 64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            ConvLayer(64, 1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid()
            # output shape:-> [bs, 1, 32, 32]
        )

        # binary threshold
        # self.threshold = nn.Sequential(
        #     nn.Linear(in_features=1 * 32 * 32, out_features=32),
        #     nn.ReLU(),
        #     nn.Linear(in_features=32, out_features=1),
        #     nn.Sigmoid()
        # )
        self.threshold = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 8, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 1 , kernel_size=4),
            nn.Sigmoid(),
        )

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    def binary(self, x):
        # xs = x.view(-1, 1 * 32 * 32)
        xs = self.threshold(x)
        # xs.shape:-> [bs, 1]; x.shape:-> [bs, 1, 32, 32]
        # xs = xs.unsqueeze(-1).repeat_interleave(repeats=32, dim=1).repeat_interleave(repeats=32, dim=2)
        # xs = xs.unsqueeze(1)
        xs = torch.sign(x - xs)
        x = torch.relu(xs)

        return x

    def forward(self, x):
        x = self.feature(x)
        x = self.stn(x)
        x = self.binary(x)

        return x


class ConvNetEfficientSTNet(nn.Module):
    def __init__(self):
        super(ConvNetEfficientSTNet, self).__init__()
        # Initial convolution layers
        self.feature = nn.Sequential(
            # input shape:-> [bs, 3, 128, 128]
            ConvLayer(3, 32, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            ConvLayer(32, 64, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            ConvLayer(64, 128, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            # EfficientNet MBlock
            EfficientVSResidual(width_coefficient=1, depth_coefficient=1, drop_connect_rate=0.),
            ConvLayer(128, 64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            ConvLayer(64, 1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU()
            # output shape:-> [bs, 1, 32, 32]
        )

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    def forward(self, x):
        x = self.feature(x)
        x = self.stn(x)

        return x


class ConvNetEfficientSTNetConvNet(nn.Module):
    def __init__(self):
        super(ConvNetEfficientSTNetConvNet, self).__init__()
        # Initial convolution layers
        # input shape:-> [bs, 3, 128, 128]
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=128)

        # EfficientNet MBlock
        self.efficient = EfficientVSResidual(width_coefficient=1, depth_coefficient=1, drop_connect_rate=0.)
        # output shape:-> [bs, 128, 32, 32]

        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.conv5 = ConvLayer(64, 1, kernel_size=1, stride=1, bias=False)
        self.bn5 = nn.BatchNorm2d(num_features=1)
        # output shape:-> [bs, 1, 32, 32]

        self.localization = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.efficient(x)
        x = self.stn(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        return x


class ConvNetEfficientSTNetBinaryConvNet(nn.Module):
    def __init__(self):
        super(ConvNetEfficientSTNetBinaryConvNet, self).__init__()
        # Initial convolution layers
        # input shape:-> [bs, 3, 128, 128]
        self.conv1 = ConvLayer(3, 32, kernel_size=5, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=128)

        # EfficientNet MBlock
        self.efficient = EfficientVSResidual(width_coefficient=1, depth_coefficient=1, drop_connect_rate=0.)
        # output shape:-> [bs, 128, 32, 32]

        self.conv4 = ConvLayer(128, 64, kernel_size=3, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.conv5 = ConvLayer(64, 1, kernel_size=1, stride=1, bias=False)
        self.bn5 = nn.BatchNorm2d(num_features=1)
        # output shape:-> [bs, 1, 32, 32]

        self.localization = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.threshold = nn.Sequential(
            nn.Conv2d(128, 128, groups=128, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, groups=128, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, groups=128, kernel_size=4),
            nn.Sigmoid()
        )

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    def binary(self, x):
        # x.shape:-> [bs, 128, 32, 32]
        # xs.shape:-> [bs, 128, 1, 1]
        xs = self.threshold(x)
        xs = torch.sign(x - xs)
        x = torch.relu(xs)
        return x

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.efficient(x)
        x = self.stn(x)
        x = self.binary(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        return x
