"""Single backbone factory shared by training, evaluation and deployment.

`segresnet` reproduces the current training architecture. Early pilot checkpoints
used shallower blocks with dropout; select `segresnet_legacy` explicitly for those
weights. An old checkpoint's config alone may not identify that historical change.
Always load state dictionaries strictly; never discard incompatible weights.
"""
from __future__ import annotations

from monai.networks.nets import AttentionUnet, BasicUNetPlusPlus, SegResNet, UNet


def build_backbone(config, device=None):
    name = config.get("backbone", "segresnet").lower()
    filters = config.get("backbone_filters", 32)
    if name in ("segresnet", "segresnet_legacy"):
        architecture = (dict(blocks_down=(1, 1, 2, 2), blocks_up=(1, 1, 1),
                             dropout_prob=.1) if name == "segresnet_legacy" else {})
        model = SegResNet(spatial_dims=3, in_channels=4, out_channels=4,
                          init_filters=filters, **architecture)
    elif name == "caranet3d":
        from src.models.caranet3d import CaraNet3D
        model = CaraNet3D(in_channels=4, out_channels=4)
    elif name == "unet":
        model = UNet(spatial_dims=3, in_channels=4, out_channels=4,
                     channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2))
    elif name == "attention_unet":
        model = AttentionUnet(spatial_dims=3, in_channels=4, out_channels=4,
                              channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2))
    elif name in ("unet++", "unetplusplus"):
        model = BasicUNetPlusPlus(spatial_dims=3, in_channels=4, out_channels=4,
                                  features=(32, 32, 64, 128, 256, 32))
    elif name in ("unet3plus", "unet+++"):
        from src.models.unet3plus3d import UNet3Plus3D
        model = UNet3Plus3D(in_channels=4, out_channels=4, channels=(32, 64, 128, 256))
    elif name == "segresnet_cfp":
        from src.models.segresnet_cfp import SegResNetCFP
        model = SegResNetCFP(in_channels=4, out_channels=4, init_filters=filters)
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return model.to(device) if device is not None else model
