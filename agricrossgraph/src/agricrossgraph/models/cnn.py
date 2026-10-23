import timm


def build_cnn(cfg, num_classes: int):
    m = cfg.model
    return timm.create_model(m.backbone, pretrained=m.pretrained, num_classes=num_classes, drop_rate=m.drop_rate)
