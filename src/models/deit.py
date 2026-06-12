import timm


def deit(num_classes):
    model = timm.create_model(
        'deit_tiny_patch16_224.fb_in1k',
        pretrained=True,
        num_classes=num_classes,
    )

    model.name = "deit"
    data_config = timm.data.resolve_model_data_config(model)
    model.transforms = timm.data.create_transform(**data_config, is_training=False)
    return model


