from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import get_cifar10_datasets
from .models import create_model


@dataclass
class TrainConfig:
    data_dir: str = "data"
    batch_size: int = 128
    num_epochs: int = 1
    lr: float = 0.1
    weight_decay: float = 5e-4
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy(logits.detach(), targets)

    n = len(loader)
    return running_loss / n, running_acc / n


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="eval", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            running_loss += loss.item()
            running_acc += accuracy(logits, targets)
    n = len(loader)
    return running_loss / n, running_acc / n


def main(cfg: TrainConfig = TrainConfig()) -> None:
    device = cfg.device
    train_ds, test_ds = get_cifar10_datasets(cfg.data_dir)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay
    )

    for epoch in range(cfg.num_epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        print(
            f"epoch={epoch+1}/{cfg.num_epochs} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={te_loss:.4f} val_acc={te_acc:.4f}"
        )

    # Save a checkpoint
    ckpt_path = "models/baseline_resnet18.pt"
    torch.save({"model_state": model.state_dict()}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
