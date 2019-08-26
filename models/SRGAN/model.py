import torch
import torch.nn as nn
import torch.nn.functional as F
from models.torchdiffeq.conv import ODEfunc, ODEBlock
from models.torchdiffeq.augmented_conv import ConvODEFunc as AugODEFunc, ODEBlock as AugODEBlock
import torch.nn.init as init


def swish(x):
    return x * F.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel, out_channels, stride):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x


class UpsampleBlock(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        return swish(self.shuffler(self.conv(x)))


def initialize_weights(net, scale=1):
    if not isinstance(net, list):
        net = [net]
    for net in net:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.ModuleList):
                for submodule in m:
                    initialize_weights(submodule)


class DiffGenerator(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor, num_channels=3, base_filter=16):
        super(DiffGenerator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor
        self.differential_blocks = ODEBlock(ODEfunc(base_filter, nb=n_residual_blocks, normalization=False))
        self.conv_first = nn.Conv2d(num_channels, base_filter, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(base_filter, 3, kernel_size=3, stride=1, padding=1)
        initialize_weights(self)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=self.upsample_factor, mode='bilinear', align_corners=False)
        out = swish(self.conv_first(out))
        out = self.differential_blocks(out)
        out = swish(out)
        out = self.conv_last(out)
        return out


class AugmentedDiffGenerator(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor, num_channels=3, base_filter=16):
        super(AugmentedDiffGenerator, self).__init__()
        self.activation = torch.nn.ReLU()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor
        augment_dim = base_filter // 4
        self.differential_blocks = AugODEBlock(
            AugODEFunc(img_size=(base_filter, 64, 64), num_filters=base_filter, augment_dim=augment_dim,
                       time_dependent=True), is_conv=True, adjoint=True, tol=5e-3)
        self.conv_first = nn.Conv2d(num_channels, base_filter, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(base_filter+augment_dim, 3, kernel_size=3, stride=1, padding=1)
        initialize_weights(self)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=self.upsample_factor, mode='bilinear', align_corners=False)
        out = self.activation(self.conv_first(out))
        out = self.differential_blocks(out)
        out = self.activation(out)
        out = self.conv_last(out)
        return out


class Generator(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor, num_channels=3, base_filter=64):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(num_channels, base_filter, kernel_size=9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1),
                            ResidualBlock(in_channels=base_filter, out_channels=base_filter, kernel=3, stride=1))

        self.conv2 = nn.Conv2d(base_filter, base_filter, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filter)

        for i in range(self.upsample_factor // 2):
            self.add_module('upsample' + str(i + 1), UpsampleBlock(base_filter))

        self.conv3 = nn.Conv2d(base_filter, num_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = swish(self.conv1(x))

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)

        x = self.bn2(self.conv2(y)) + x

        for i in range(self.upsample_factor // 2):
            x = self.__getattr__('upsample' + str(i + 1))(x)

        return self.conv3(x)

    def weight_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class Discriminator(nn.Module):
    def __init__(self, num_channels=3, base_filter=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, base_filter, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(base_filter, base_filter * 2, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(base_filter * 2)
        self.conv5 = nn.Conv2d(base_filter * 2, base_filter * 4, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(base_filter * 4)
        self.conv7 = nn.Conv2d(base_filter * 4, base_filter * 8, kernel_size=3, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(base_filter * 8)

        # Replaced original paper FC layers with FCN
        self.conv9 = nn.Conv2d(base_filter * 8, num_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = swish(self.conv1(x))

        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn7(self.conv7(x)))

        x = self.conv9(x)
        return torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)

    def weight_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
