import torch
import wandb
import csv
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from typing import Tuple
from sklearn.metrics import f1_score

CLASS_NAMES = ["can", "cardboard", "glass", "paper", "plastic", "trash"]


def train_step(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device
) -> Tuple[float, float]:
    model.train()
    train_acc = 0
    train_loss = 0
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = y_pred.argmax(dim=1)
        train_acc += (y_pred_class == y).sum().item()

        wandb.log({"train/batch_loss": loss.item()})

    train_loss = train_loss / len(data_loader)
    train_acc = train_acc / len(data_loader.dataset)
    return train_loss, train_acc


def eval_step(
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device
) -> Tuple[float, float, float, list, list]:
    model.eval()
    test_loss, test_acc = 0, 0
    all_preds = []
    all_labels = []
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            y_pred_class = y_pred.argmax(dim=1)
            test_acc += (y_pred_class == y).sum().item()
            all_preds.extend(y_pred_class.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_loss = test_loss / len(data_loader)
    test_acc = test_acc / len(data_loader.dataset)
    final_f1 = f1_score(all_labels, all_preds, average="macro")
    return test_loss, test_acc, final_f1, all_preds, all_labels


def train(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        device: torch.device,
        run: wandb.sdk.wandb_run.Run
):
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "f1_macro": []
    }

    csv_dir = Path.cwd() / "confusion_data"
    csv_dir.mkdir(exist_ok=True)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            data_loader=train_loader,
            device=device
        )
        val_loss, val_acc, f1_macro, all_preds, all_labels = eval_step(
            model=model,
            loss_fn=loss_fn,
            data_loader=val_loader,
            device=device
        )

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f} | "
            f"f1_macro: {f1_macro:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["f1_macro"].append(f1_macro)

        # epoch-level scalars
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/f1_macro": f1_macro,
        })

        # visual confusion matrix
        wandb.log({
            "val/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels,
                preds=all_preds,
                class_names=CLASS_NAMES
            )
        })

        # CSV artifact
        csv_path = csv_dir / f"{run.name}_epoch{epoch}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "true_label", "predicted_label"])
            writer.writerows([[epoch, t, p] for t, p in zip(all_labels, all_preds)])

        artifact = wandb.Artifact(name=f"{run.name}_confusion", type="confusion_data")
        artifact.add_file(str(csv_path))
        wandb.log_artifact(artifact)

    return results