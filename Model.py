import torch
from torch import nn


# ! TODO Stride 사이즈 먹이기
# ! TODO 모듈단위 테스트
# ! TODO Squeeze size 결정하기


class IdentityModule(nn.Module):
    def __init__(self):
        super(IdentityModule, self).__init__()

    def forward(self, x):
        return x


class SqueezeExciteModule(nn.Module):
    def __init__(self, expand_size):
        super(SqueezeExciteModule, self).__init__()

        self.se_0_0 = nn.AdaptiveAvgPool2d(output_size=1)
        self.se_0_1 = nn.Flatten()

        self.se_1_0 = nn.Linear(in_features=expand_size, out_features=expand_size)
        self.se_1_1 = nn.ReLU(inplace=True)

        self.se_2_0 = nn.Linear(in_features=expand_size, out_features=expand_size)
        self.se_2_1 = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        x = self.se_0_0(x)
        x = self.se_0_1(x)

        x = self.se_1_0(x)
        x = self.se_1_1(x)

        x = self.se_2_0(x)
        x = self.se_2_1(x)
        x = torch.unsqueeze(x, -1)
        x = torch.unsqueeze(x, -1)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dw_kernel_size, expand_size, squeeze_excite,
                 nonlinearity, stride):
        super(Bottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_size = expand_size
        self.squeeze_excite = squeeze_excite
        self.stride = stride
        self.dw_kernel_size = dw_kernel_size

        if nonlinearity == 'hardswish':
            self.Nonliearity = nn.Hardswish
        elif nonlinearity == 'relu':
            self.Nonliearity = nn.ReLU
        else:
            raise RuntimeError("No such nonlinearity!")

        # 1x1 Conv2d + NL
        self.bottleneck_0_0 = nn.Conv2d(in_channels=in_channels, out_channels=expand_size, kernel_size=(1, 1),
                                        bias=False)
        self.bottleneck_0_1 = nn.BatchNorm2d(num_features=expand_size)
        self.bottleneck_0_2 = self.Nonliearity(inplace=True)

        # Dwise + NL
        self.bottleneck_1_0 = nn.Conv2d(in_channels=expand_size, out_channels=expand_size,
                                        kernel_size=self.dw_kernel_size,
                                        stride=self.stride, padding=self.dw_kernel_size[0] // 2, groups=expand_size,
                                        bias=False)
        self.bottleneck_1_1 = nn.BatchNorm2d(num_features=expand_size)

        # Squeeze-Excite
        if self.squeeze_excite:
            self.squeeze_excite_0 = SqueezeExciteModule(
                expand_size=expand_size
            )
        else:
            self.squeeze_excite_0 = IdentityModule()

        # Final 1x1 Conv2d
        self.bottleneck_final_0 = nn.Conv2d(in_channels=expand_size, out_channels=out_channels, kernel_size=(1, 1),
                                            bias=False)
        self.bottleneck_final_1 = nn.BatchNorm2d(num_features=out_channels)

        # Downsampling first layer
        self.bottleneck_final_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=(1, 1), stride=self.stride, bias=False)

    def forward(self, x):
        x_0 = self.bottleneck_0_0(x)
        x_0 = self.bottleneck_0_1(x_0)
        x_0 = self.bottleneck_0_2(x_0)

        x_0 = self.bottleneck_1_0(x_0)
        x_0 = self.bottleneck_1_1(x_0)

        x_1 = self.squeeze_excite_0(x_0)
        x_0 = x_0 * x_1

        x_0 = self.bottleneck_final_0(x_0)
        x_0 = self.bottleneck_final_1(x_0)
        x_b = self.bottleneck_final_2(x)
        return x_0.add(x_b)


class MobileNetV3(nn.Module):
    def __init__(self, size, width_mult=1.0):
        super(MobileNetV3, self).__init__()

        if size != 'small':
            raise RuntimeError("Not implemented except small model")

        self.conv_0_0 = nn.Conv2d(in_channels=3, out_channels=int(16 * width_mult),
                                  kernel_size=(3, 3), stride=2, padding=3 // 2, bias=False)
        self.conv_0_1 = nn.BatchNorm2d(num_features=int(16 * width_mult))
        self.conv_0_2 = nn.Hardswish(inplace=True)

        self.conv_1_0 = Bottleneck(in_channels=int(16 * width_mult),
                                   out_channels=int(16 * width_mult), dw_kernel_size=(3, 3),
                                   expand_size=16, squeeze_excite=True, nonlinearity='relu', stride=2)
        self.conv_2_0 = Bottleneck(in_channels=int(16 * width_mult),
                                   out_channels=int(24 * width_mult), dw_kernel_size=(3, 3),
                                   expand_size=72, squeeze_excite=False, nonlinearity='relu', stride=2)
        self.conv_3_0 = Bottleneck(in_channels=int(24 * width_mult),
                                   out_channels=int(24 * width_mult), dw_kernel_size=(3, 3),
                                   expand_size=88, squeeze_excite=False, nonlinearity='relu', stride=1)
        self.conv_4_0 = Bottleneck(in_channels=int(24 * width_mult),
                                   out_channels=int(40 * width_mult), dw_kernel_size=(5, 5),
                                   expand_size=96, squeeze_excite=True, nonlinearity='hardswish', stride=2)
        self.conv_5_0 = Bottleneck(in_channels=int(40 * width_mult),
                                   out_channels=int(40 * width_mult), dw_kernel_size=(5, 5),
                                   expand_size=240, squeeze_excite=True, nonlinearity='hardswish', stride=1)
        self.conv_6_0 = Bottleneck(in_channels=int(40 * width_mult),
                                   out_channels=int(40 * width_mult), dw_kernel_size=(5, 5),
                                   expand_size=240, squeeze_excite=True, nonlinearity='hardswish', stride=1)
        self.conv_7_0 = Bottleneck(in_channels=int(40 * width_mult),
                                   out_channels=int(48 * width_mult), dw_kernel_size=(5, 5),
                                   expand_size=120, squeeze_excite=True, nonlinearity='hardswish', stride=1)
        self.conv_8_0 = Bottleneck(in_channels=int(48 * width_mult),
                                   out_channels=int(48 * width_mult), dw_kernel_size=(5, 5),
                                   expand_size=144, squeeze_excite=True, nonlinearity='hardswish', stride=1)
        self.conv_9_0 = Bottleneck(in_channels=int(48 * width_mult),
                                   out_channels=int(96 * width_mult), dw_kernel_size=(5, 5),
                                   expand_size=288, squeeze_excite=True, nonlinearity='hardswish', stride=2)
        self.conv_10_0 = Bottleneck(in_channels=int(96 * width_mult),
                                    out_channels=int(96 * width_mult), dw_kernel_size=(5, 5),
                                    expand_size=576, squeeze_excite=True, nonlinearity='hardswish', stride=1)
        self.conv_11_0 = Bottleneck(in_channels=int(96 * width_mult),
                                    out_channels=int(96 * width_mult), dw_kernel_size=(5, 5),
                                    expand_size=576, squeeze_excite=True, nonlinearity='hardswish', stride=1)

        self.conv_12_0 = nn.Conv2d(in_channels=int(96 * width_mult), out_channels=int(576 * width_mult),
                                   kernel_size=(1, 1), bias=False)
        self.conv_12_1 = nn.Hardswish(inplace=True)
        self.conv_12_2 = nn.BatchNorm2d(num_features=int(576 * width_mult))

        self.features = nn.Sequential(
            self.conv_0_0,
            self.conv_0_1,
            self.conv_0_2,
            self.conv_1_0,
            self.conv_2_0,
            self.conv_3_0,
            self.conv_4_0,
            self.conv_5_0,
            self.conv_6_0,
            self.conv_7_0,
            self.conv_8_0,
            self.conv_9_0,
            self.conv_10_0,
            self.conv_11_0,
            self.conv_12_0,
            self.conv_12_1,
            self.conv_12_2
        )

    def forward(self, x):
        x = self.features(x)
        return x
