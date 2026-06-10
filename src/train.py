"""Training entry point."""

import torch
import wandb
from src import engine
from src.models import MODELS
from src.data_test import get_loaders
from src.utils import save_model
from dotenv import load_dotenv

load_dotenv()

NUM_CLASSES = 6
BATCH_SIZE = 32
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_all():
    for model_fn, epochs in MODELS:
        created_model = model_fn(num_classes=NUM_CLASSES)
        created_model.to(DEVICE)

        run = wandb.init(
            entity="trash-busters",
            project="trash-busters",
            name=created_model.name,
            config={
                "model": created_model.name,
                "dataset": "trashnet",
                "num_classes": NUM_CLASSES,
                "epochs": epochs,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "optimizer": "Adam",
                "loss": "CrossEntropyLoss",
                "device": str(DEVICE),
            }
        )

        train_loader, val_loader = get_loaders(created_model.transforms, BATCH_SIZE)
        wandb.watch(created_model, log="all", log_freq=100)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=created_model.parameters(), lr=LR)

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
