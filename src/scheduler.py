import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Vision model (ResNet, EfficientNet, etc.)
model = ...
train_loader = ...  # Main training dataset loader
val_loader = ...  # Complete validation loader (used at end of epoch)
proxy_val_loader = ...  # 5-10% stratified subset for fast sub-epoch checks

# Standard evaluating performance in multi-class image classification
criterion = nn.CrossEntropyLoss()

# Using AdamW optimizer with a baseline starting learning rate of 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# ReduceLROnPlateau Scheduler Initialized with Baseline Values
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",  # 'min' because we want to minimize validation loss
    factor=0.5,  # Halving the LR on a plateau provides a noticeable adaptation
    patience=20,  # Scale patience up to 20 because we step sub-epochly (every N batches)
    threshold=1e-4,  # Ignores minor noisy decimal shifts to guarantee meaningful progress
    cooldown=2,  # Grace period of epochs to allow the training curve to stabilize
    min_lr=1e-6,  # Absolute floor; dropping below 1e-6 yields diminishing returns
)

# Cost-effective diagnostic & early stopping trackers
CHECK_EVERY_N_BATCHES = 500  # Evaluate loop frequently instead of waiting for full epoch hours
consecutive_lr_drops = 0  # Counter tracking consecutive reductions without a breakthrough
epochs = 50

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        # Standard forward pass and optimizer optimization
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Sub-epoch evaluation inner loop to check progress faster
        if (i + 1) % CHECK_EVERY_N_BATCHES == 0:
            model.eval()
            proxy_val_loss = 0.0

            # Fast validation via proxy sub-sampling
            with torch.no_grad():
                for val_images, val_labels in proxy_val_loader:
                    val_outputs = model(val_images)
                    v_loss = criterion(val_outputs, val_labels)
                    proxy_val_loss += v_loss.item()

            avg_proxy_loss = proxy_val_loss / len(proxy_val_loader)

            # Track the current learning rate before stepping the scheduler
            old_lr = optimizer.param_groups[0]["lr"]

            # Feed sub-sample validation loss into scheduler to adapt quickly
            scheduler.step(avg_proxy_loss)

            # Fetch the updated learning rate to run automated health checks
            new_lr = optimizer.param_groups[0]["lr"]

            # Automated health checks & custom early stopping
            if new_lr < old_lr:
                consecutive_lr_drops += 1
                print(
                    f"[Iteration {i+1}] Plateau detected. Learning rate dropped to {new_lr}"
                )

                # Stop if we hit our maximum patience for consecutive drops without breakthrough
                if consecutive_lr_drops >= 3:
                    print(
                        "Early stopping triggered: 3 consecutive LR reductions without a loss breakthrough."
                    )
                    print(
                        "Pipeline health check needed (verify labels, image preprocessing, or architecture)."
                    )
                    break
            else:
                consecutive_lr_drops = 0

            # Stop if the scheduler hits its absolute floor (min_lr)
            if new_lr <= scheduler.min_lrs[0]:
                print(
                    "Early stopping triggered: Learning rate reached the minimum threshold floor."
                )
                break

            model.train()  # Reset back to training mode

    # Break out of outer epoch loop if inner loop triggered early stopping
    if (
        consecutive_lr_drops >= 3
        or optimizer.param_groups[0]["lr"] <= scheduler.min_lrs[0]
    ):
        break

    # Run a complete validation pass on the full dataset only once at the end of an epoch
    model.eval()
    full_val_loss = 0.0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_outputs = model(val_images)
            v_loss = criterion(val_outputs, val_labels)
            full_val_loss += v_loss.item()

    print(
        f"Epoch {epoch+1} complete. Comprehensive Val Loss: {full_val_loss / len(val_loader)}"
    )
