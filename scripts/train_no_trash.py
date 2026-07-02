"""Ablation entry point: train baseline + convnext WITHOUT the 'trash' class.

Reuses the existing pipeline (``src.train.train_all``) via backward-compatible
overrides, so it does not change any default behaviour:

  * trains only ``baseline`` (resnet18) and ``convnext_tiny``
  * drops the ``trash`` class from every split -- train/val/test on the combined
    set AND the held-out custom "wild" eval -- giving a 5-class classifier
  * keeps the custom wild dataset as a held-out evaluation set (as today)
  * tags runs / checkpoints with ``no_trash`` so they never collide with the
    default 6-class runs

Run from the repo root:

    python -m scripts.train_no_trash
"""

from src.train import train_all
from src.engine import CLASS_NAMES
from src.utils import load_config, seed_everything

# Class to exclude, and the models to train (names as registered in MODEL_REGISTRY).
EXCLUDED_CLASS = "trash"
MODELS = ["resnet18", "convnext_tiny"]  # resnet18.name == "baseline"


def main() -> None:
    config = load_config()

    # Match the default pipeline's reproducibility behaviour.
    seed_everything(config["project"]["random_seed"])

    # 'trash' is the last class, so dropping it leaves labels 0..4 contiguous.
    class_names = [name for name in CLASS_NAMES if name != EXCLUDED_CLASS]

    train_all(
        model_names=MODELS,
        epochs=config["training"]["epochs"],
        run_name="no_trash",
        num_classes=len(class_names),
        exclude_classes=[EXCLUDED_CLASS],
        class_names=class_names,
        dataset_tag=f"{config['training']['wandb']['dataset']}_no_trash",
    )


if __name__ == "__main__":
    main()
