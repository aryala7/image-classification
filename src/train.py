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
EPOCHS = 30 # can later be exchanged as arguments in MODELS e.g.: MODELS = [(resnet18, 15), (convnext, 50), ...]
BATCH_SIZE = 32 # needs to be adjusted
LR = 0.001 # can later be exchanged for a LR scheduler
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_all():
    for model_fn in MODELS: # can later be exchanged as "for model_fn, epochs in MODELS"
        created_model = model_fn(num_classes=NUM_CLASSES)
        created_model.to(DEVICE)

        # each model gets its own W&B run
        run = wandb.init(
            entity="trash-busters",
            project="trash-busters",        # replace "garbage-classification"
            name=created_model.name,
            config={...}
        )   


        # dataset setup goes here
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
            epochs=EPOCHS,
            device=DEVICE,
            run=run
        )

        save_model(
            model=created_model,
            target_dir="models",
            model_name=f"{created_model.name}_{EPOCHS}epochs.pth",
            log_artifact=True
        )

        run.finish()
        print("-" * 50 + "\n")


if __name__ == "__main__":
    train_all()