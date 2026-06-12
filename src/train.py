"""Training entry point."""

import torch
import wandb
from src import engine
from src.models import MODELS
from src.data_test import get_loaders
from src.utils import save_model, load_config
from dotenv import load_dotenv

load_dotenv()

config = load_config()
training_config = config["training"]

NUM_CLASSES = training_config["num_classes"]
BATCH_SIZE = training_config["batch_size"]
LR = training_config["lr"]
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


def train_all():
    for model_fn, epochs in MODELS:
        created_model = model_fn(num_classes=NUM_CLASSES)
        created_model.to(DEVICE)

        run = wandb.init(
            entity=training_config["wandb"]["entity"],
            project=training_config["wandb"]["project"],
            name=created_model.name,
            config={
                "model": created_model.name,
                "dataset": training_config["wandb"]["dataset"],
                "num_classes": NUM_CLASSES,
                "epochs": epochs,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "optimizer": training_config["optimizer"],
                "loss": training_config["loss"],
                "device": str(DEVICE),
            }
        )

        train_loader, val_loader = get_loaders(created_model.transforms, BATCH_SIZE)
        wandb.watch(created_model, log="all", log_freq=100)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params=created_model.parameters(), lr=LR)

        engine.train(
            model=created_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            device=DEVICE,
            run=run
        )

        save_model(
            model=created_model,
            target_dir="models",
            model_name=f"{created_model.name}_{epochs}epochs.pth",
            log_artifact=True
        )

        run.finish()
        print("-" * 50 + "\n")


if __name__ == "__main__":
    train_all()
