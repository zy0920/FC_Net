import logging

from cvpods.layers import ShapeSpec
from cvpods.modeling.anchor_generator import ShiftGenerator
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone.fpn import build_retinanet_resnet_fpn_p5_backbone

from fcos import FCOS


import torch.nn as nn
from torchvision.models import ResNet

#ԭnet.py
def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = build_retinanet_resnet_fpn_p5_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_shift_generator(cfg, input_shape):

    return ShiftGenerator(cfg, input_shape)


def build_model(cfg):

    cfg.build_backbone = build_backbone
    cfg.build_shift_generator = build_shift_generator

    model = FCOS(cfg)
    #model = ResNet(FcaBottleneck, [3, 4, 23, 3], num_classes=70)
    #model.avgpool = nn.AdaptiveAvgPool2d(1)

    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    return model
