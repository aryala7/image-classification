import timm


def mlp_mixer(num_classes):
    model = timm.create_model(
        'mixer_b16_224.miil_in21k_ft_in1k',
        pretrained=True,
        num_classes=num_classes,
    )

    model.name = "mlp_mixer"
    data_config = timm.data.resolve_model_data_config(model)
    model.transforms = timm.data.create_transform(**data_config, is_training=False)
    return model


