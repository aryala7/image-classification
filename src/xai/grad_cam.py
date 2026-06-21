import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def generate_gradcam(
        model,
        image_path,
        target_layer,
        transform,
        class_idx=None,
        device="cpu",
        output_path="grad_cam.jpg"
):
    model = model.to(device)
    model.eval()

    image = Image.open(image_path)
    tensor = transform(image).unsqueeze(0).to(device)
    rgb_image = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

    with torch.no_grad():
        outputs = model(tensor)
        predicted_class = outputs.argmax(dim=1).item()
    if class_idx is None:
        class_idx = predicted_class

    targets = [ClassifierOutputTarget(class_idx)]

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )
    Image.fromarray(visualization).save(output_path)
    return {
        "predicted_class": predicted_class,
        "class_idx": class_idx,
        "output_path": output_path
    }

    pass

def generate_gradcam_for_samples(
    model,
    samples,
    target_layer,
    transform,
    class_names,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir="xai_results",
):
    import os
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for i, sample in enumerate(samples):
        image_path = sample["image_path"]
        true_label = sample["label"]

        output_path = f"{output_dir}/sample_{i}_gradcam.jpg"

        result = generate_gradcam(
            model=model,
            image_path=image_path,
            target_layer=target_layer,
            transform=transform,
            class_idx=None,
            device=device,
            output_path=output_path,
        )

        predicted_label = result["predicted_class"]

        results.append({
            "image_path": image_path,
            "true_label": class_names[true_label],
            "predicted_label": class_names[predicted_label],
            "gradcam_path": output_path,
            "correct": true_label == predicted_label,
        })

    return results