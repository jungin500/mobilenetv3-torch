import torch
from torch import nn

#! TODO Stride 사이즈 먹이기
#! TODO 모듈단위 테스트
#! TODO Squeeze size 결정하기


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
        self.se_1_1 = nn.ReLU(inplace=False)

        self.se_2_0 = nn.Linear(in_features=expand_size, out_features=expand_size)
        self.se_2_1 = nn.Hardsigmoid(inplace=False)

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
    def __init__(self, in_feature_size, in_channels, out_channels, dw_kernel_size, expand_size, squeeze_excite, nonlinearity, stride):
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
        self.bottleneck_0_0 = nn.Conv2d(in_channels=in_channels, out_channels=expand_size, kernel_size=(1, 1), bias=False)
        self.bottleneck_0_1 = nn.BatchNorm2d(num_features=expand_size)
        self.bottleneck_0_2 = self.Nonliearity()

        # Dwise + NL
        self.bottleneck_1_0 = nn.Conv2d(in_channels=expand_size, out_channels=expand_size, kernel_size=self.dw_kernel_size,
                                    stride=self.stride, padding=self.dw_kernel_size[0] // 2, groups=expand_size, bias=False)
        self.bottleneck_1_1 = nn.BatchNorm2d(num_features=expand_size)

        # Squeeze-Excite
        if self.squeeze_excite:
            self.squeeze_excite_0 = SqueezeExciteModule(
                expand_size=expand_size
            )
        else:
            self.squeeze_excite_0 = IdentityModule()

        # Final 1x1 Conv2d
        self.bottleneck_final_0 = nn.Conv2d(in_channels=expand_size, out_channels=out_channels, kernel_size=(1, 1), bias=False)
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
    def __init__(self, size, out_features):
        super(MobileNetV3, self).__init__()

        if size != 'small':
            raise RuntimeError("Not implemented except small model")

        self.conv_0_0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=2, padding=3 // 2, bias=False)
        self.conv_0_1 = nn.BatchNorm2d(num_features=16)
        self.conv_0_2 = nn.Hardswish()

        self.conv_1_0 = Bottleneck(in_feature_size=112, in_channels=16, out_channels=16, dw_kernel_size=(3, 3),
                                   expand_size=16, squeeze_excite=True, nonlinearity='relu', stride=2)
        self.conv_2_0 = Bottleneck(in_feature_size=56, in_channels=16, out_channels=24,  dw_kernel_size=(3, 3),
                                   expand_size=72, squeeze_excite=False, nonlinearity='relu', stride=2)
        self.conv_3_0 = Bottleneck(in_feature_size=28, in_channels=24, out_channels=24,  dw_kernel_size=(3, 3),
                                   expand_size=88, squeeze_excite=False, nonlinearity='relu', stride=1)
        self.conv_4_0 = Bottleneck(in_feature_size=28, in_channels=24, out_channels=40,  dw_kernel_size=(5, 5),
                                   expand_size=96, squeeze_excite=True, nonlinearity='hardswish', stride=2)
        self.conv_5_0 = Bottleneck(in_feature_size=14, in_channels=40, out_channels=40,  dw_kernel_size=(5, 5),
                                   expand_size=240, squeeze_excite=True, nonlinearity='hardswish', stride=1)
        self.conv_6_0 = Bottleneck(in_feature_size=14, in_channels=40, out_channels=40,  dw_kernel_size=(5, 5),
                                   expand_size=240, squeeze_excite=True, nonlinearity='hardswish', stride=1)
        self.conv_7_0 = Bottleneck(in_feature_size=14, in_channels=40, out_channels=48, dw_kernel_size=(5, 5),
                                   expand_size=120, squeeze_excite=True, nonlinearity='hardswish', stride=1)
        self.conv_8_0 = Bottleneck(in_feature_size=14, in_channels=48, out_channels=48, dw_kernel_size=(5, 5),
                                   expand_size=144, squeeze_excite=True, nonlinearity='hardswish', stride=1)
        self.conv_9_0 = Bottleneck(in_feature_size=14, in_channels=48, out_channels=96, dw_kernel_size=(5, 5),
                                   expand_size=288, squeeze_excite=True, nonlinearity='hardswish', stride=2)
        self.conv_10_0 = Bottleneck(in_feature_size=7, in_channels=96, out_channels=96, dw_kernel_size=(5, 5),
                                   expand_size=576, squeeze_excite=True, nonlinearity='hardswish', stride=1)
        self.conv_11_0 = Bottleneck(in_feature_size=7, in_channels=96, out_channels=96, dw_kernel_size=(5, 5),
                                   expand_size=576, squeeze_excite=True, nonlinearity='hardswish', stride=1)

        self.conv_12_0 = nn.Conv2d(in_channels=96, out_channels=576, kernel_size=(1, 1), bias=False)
        self.conv_12_1 = nn.Hardswish()
        self.conv_12_2 = nn.BatchNorm2d(num_features=576)

        self.conv_13_0 = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv_14_0 = nn.Conv2d(in_channels=576, out_channels=1024, kernel_size=(1, 1), bias=False)
        self.conv_14_1 = nn.Hardswish()

        self.conv_15_0 = nn.Conv2d(in_channels=1024, out_channels=out_features, kernel_size=(1, 1), bias=False)


    def forward(self, x):
        x = self.conv_0_0(x)
        x = self.conv_0_1(x)
        x = self.conv_0_2(x)

        x = self.conv_1_0(x)
        x = self.conv_2_0(x)
        x = self.conv_3_0(x)
        x = self.conv_4_0(x)
        x = self.conv_5_0(x)
        x = self.conv_6_0(x)
        x = self.conv_7_0(x)
        x = self.conv_8_0(x)
        x = self.conv_9_0(x)
        x = self.conv_10_0(x)
        x = self.conv_11_0(x)
        x = self.conv_12_0(x)
        x = self.conv_12_1(x)
        x = self.conv_12_2(x)
        x = self.conv_13_0(x)
        x = self.conv_14_0(x)
        x = self.conv_14_1(x)
        x = self.conv_15_0(x)
        x = torch.flatten(x, start_dim=1)

        return x
