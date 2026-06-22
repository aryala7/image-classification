"""Training entry point."""

import torch
import wandb
from src import engine
from src.models import MODELS
from src.data import create_training_dataloaders
from src.utils import save_model, load_config
from dotenv import load_dotenv

load_dotenv()

config = load_config()
training_config = config["training"]
data_config = config["data"]

NUM_CLASSES = training_config["num_classes"]
BATCH_SIZE = training_config["batch_size"]
LR = training_config["lr"]
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


def train_all():
    print(f"Using device: {DEVICE}")
    for model_fn, epochs in MODELS:
        print(f"Initializing model: {model_fn.__name__}")
        created_model = model_fn(num_classes=NUM_CLASSES)
        created_model.to(DEVICE)
        print(f"Training model: {created_model.name}")

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

        train_loader, val_loader, test_loader = create_training_dataloaders(
            transform=created_model.transforms,
            batch_size=BATCH_SIZE,
            data_config=data_config,
        )
        print(
            f"Dataset sizes | "
            f"train: {len(train_loader.dataset)} | "
            f"validation: {len(val_loader.dataset)} | "
            f"test: {len(test_loader.dataset)}"
        )
        wandb.watch(created_model, log="all", log_freq=100)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params=created_model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  # 'min' because we want to minimize validation loss
            factor=0.5,  # Halving the LR on a plateau provides a noticeable adaptation
            patience=3,  # Scale patience up to 3 because we step epochly (every N batches)
            threshold=1e-4,  # Ignores minor noisy decimal shifts to guarantee meaningful progress
            cooldown=2,  # Grace period of epochs to allow the training curve to stabilize
            min_lr=1e-6,  # Absolute floor; dropping below 1e-6 yields diminishing returns
        )
        engine.train(
            model=created_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            device=DEVICE,
            run=run,
            scheduler=scheduler
        )

        engine.test(
            model=created_model,
            loss_fn=loss_fn,
            test_loader=test_loader,
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
