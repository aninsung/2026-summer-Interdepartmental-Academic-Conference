import torch
import torch.nn as nn
from monai.networks.nets import SegResNet

class BasicConv3dGN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv3dGN, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        # Use GroupNorm(num_groups=8) instead of BatchNorm3d to prevent NaN with batch_size=1
        self.gn = nn.GroupNorm(8, out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.gn(self.conv(x)))

class CFPModule3DGN(nn.Module):
    # Contextual Feature Pyramid Module with GroupNorm
    def __init__(self, in_channels, out_channels):
        super(CFPModule3DGN, self).__init__()
        self.branch1 = BasicConv3dGN(in_channels, out_channels, 1)
        self.branch2 = nn.Sequential(
            BasicConv3dGN(in_channels, out_channels, 1),
            BasicConv3dGN(out_channels, out_channels, 3, padding=3, dilation=3)
        )
        self.branch3 = nn.Sequential(
            BasicConv3dGN(in_channels, out_channels, 1),
            BasicConv3dGN(out_channels, out_channels, 3, padding=5, dilation=5)
        )
        self.branch4 = nn.Sequential(
            BasicConv3dGN(in_channels, out_channels, 1),
            BasicConv3dGN(out_channels, out_channels, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv3dGN(4 * out_channels, out_channels, 3, padding=1)
        self.conv_res = BasicConv3dGN(in_channels, out_channels, 1)

    def forward(self, x):
        x0 = self.branch1(x)
        x1 = self.branch2(x)
        x2 = self.branch3(x)
        x3 = self.branch4(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x_res = self.conv_res(x)
        return x_res + x_cat

class SegResNetCFP(nn.Module):
    """
    Hybrid Backbone: SegResNet + CaraNet CFP (GroupNorm Stabilized)
    Uses SegResNet to extract robust 3D features, and refines the final boundary 
    using a stabilized Contextual Feature Pyramid (CFP) module.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 4, init_filters: int = 32):
        super().__init__()
        
        # SegResNet is configured to output `init_filters` channels (e.g. 32) instead of the final classes
        self.backbone = SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=init_filters,
            init_filters=init_filters
        )
        
        # Stabilized CFP Module with GroupNorm
        self.cfp = CFPModule3DGN(in_channels=init_filters, out_channels=init_filters // 2)
        
        # Final classifier projects to the target class number
        self.classifier = nn.Conv3d(init_filters // 2, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Feature Extraction (Robust SOTA features)
        features = self.backbone(x)
        
        # 2. Boundary Refinement (Multi-dilation context with GroupNorm)
        refined = self.cfp(features)
        
        # 3. Final Classification
        out = self.classifier(refined)
        
        return out
