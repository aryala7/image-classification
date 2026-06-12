import timm


def convnext_tiny(num_classes):
    model = timm.create_model(
        'convnext_tiny.fb_in22k_ft_in1k',
        pretrained=True,
        num_classes=num_classes,
    )

    model.name = "convnext_tiny"
    data_config = timm.data.resolve_model_data_config(model)
    model.transforms = timm.data.create_transform(**data_config, is_training=False)
    return model


