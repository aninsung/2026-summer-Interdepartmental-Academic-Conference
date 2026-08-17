import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class CFPModule(nn.Module):
    # Contextual Feature Pyramid Module (CaraNet 핵심)
    def __init__(self, in_channels, out_channels):
        super(CFPModule, self).__init__()
        self.branch1 = BasicConv2d(in_channels, out_channels, 1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, 3, padding=3, dilation=3)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, 3, padding=5, dilation=5)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channels, out_channels, 3, padding=1)
        self.conv_res = BasicConv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x0 = self.branch1(x)
        x1 = self.branch2(x)
        x2 = self.branch3(x)
        x3 = self.branch4(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x_res = self.conv_res(x)
        return x_res + x_cat

class ReverseAttention(nn.Module):
    # Reverse Attention Module (CaraNet/PraNet 핵심)
    def __init__(self, in_channels, out_channels):
        super(ReverseAttention, self).__init__()
        self.conv1 = BasicConv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x, prior_map):
        x = self.conv1(x)
        # prior_map: 이전 단계의 전역 예측 (Global Prediction)
        prior_map = F.interpolate(prior_map, size=x.size()[2:], mode='bilinear', align_corners=False)
        # 역어텐션: 확실한 부분을 지우고 경계선 및 놓친 픽셀에 집중
        reverse_map = -1 * (torch.sigmoid(prior_map)) + 1
        x = x * reverse_map
        x = self.conv2(x)
        x = self.conv3(x)
        out = self.conv_out(x)
        return out

class CaraNet(nn.Module):
    """
    Context Axial Reverse Attention Network (Simplified for BraTS)
    소형 종양 특화 스나이퍼 백본
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(CaraNet, self).__init__()
        
        # Simple Encoder (ResNet-like features)
        self.enc1 = nn.Sequential(
            BasicConv2d(in_channels, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1)
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            BasicConv2d(64, 128, 3, padding=1),
            BasicConv2d(128, 128, 3, padding=1)
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            BasicConv2d(128, 256, 3, padding=1),
            BasicConv2d(256, 256, 3, padding=1)
        )
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            BasicConv2d(256, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1)
        )

        # Contextual Feature Pyramid (CFP) Modules
        self.cfp4 = CFPModule(512, 64)
        self.cfp3 = CFPModule(256, 64)
        
        # Partial Decoder (Aggregation)
        self.pd_conv = BasicConv2d(128, 64, 3, padding=1)
        self.global_map = nn.Conv2d(64, out_channels, 1)

        # Reverse Attention (RA)
        self.ra3 = ReverseAttention(256, 64)
        self.ra2 = ReverseAttention(128, 64)

    def forward(self, x):
        # Feature Extraction
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        # High-level feature context
        cfp4 = self.cfp4(x4)
        cfp3 = self.cfp3(x3)

        # Partial Decoder (Global Map)
        pd = self.pd_conv(torch.cat((F.interpolate(cfp4, size=x3.size()[2:], mode='bilinear', align_corners=False), cfp3), 1))
        p_map = self.global_map(pd)

        # Reverse Attention cascade
        ra3_out = self.ra3(x3, p_map)
        p_map_2 = p_map + F.interpolate(ra3_out, size=p_map.size()[2:], mode='bilinear', align_corners=False)

        ra2_out = self.ra2(x2, p_map_2)
        p_map_1 = p_map_2 + F.interpolate(ra2_out, size=p_map_2.size()[2:], mode='bilinear', align_corners=False)

        # Final map upsampled to input resolution
        out = F.interpolate(p_map_1, size=x.size()[2:], mode='bilinear', align_corners=False)
        return out

def build_caranet(in_channels=1, out_channels=1):
    return CaraNet(in_channels=in_channels, out_channels=out_channels)
