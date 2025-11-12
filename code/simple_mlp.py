"""
Simple MLP example that prefers PyTorch (with MPS/Metal when available) and falls
back to scikit-learn's MLPClassifier if PyTorch is not installed.

Run:
    python code/simple_mlp.py --epochs 10

This script trains a small classifier on scikit-learn's digits dataset and prints test accuracy.
"""

from __future__ import annotations

from typing import Any

import argparse
import sys
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_sklearn(X_train, X_test, y_train, y_test, epochs: int):
    from sklearn.neural_network import MLPClassifier

    print("Using scikit-learn MLPClassifier (fallback)")
    clf = MLPClassifier(hidden_layer_sizes=(64, 64), activation="relu", max_iter=epochs, batch_size=32, random_state=42)
    start = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - start
    acc = clf.score(X_test, y_test)
    print(f"Finished (sklearn) — time: {elapsed:.2f}s  test accuracy: {acc:.4f}")


def run_torch(X_train, X_test, y_train, y_test, epochs: int, device: Any = "cpu"):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device(device)
    print(f"Using PyTorch on device: {device}")

    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).long().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)

    n_features = X_train.shape[1]
    n_classes = int(y_train.max().item()) + 1

    model = nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, n_classes)
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 32
    dataset_size = X_train.shape[0]
    steps_per_epoch = max(1, dataset_size // batch_size)

    start_time = time.time()
    acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(dataset_size)
        epoch_loss = 0.0
        for i in range(0, dataset_size, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train[idx]
            yb = y_train[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        # eval
        model.eval()
        with torch.no_grad():
            logits = model(X_test)
            preds = logits.argmax(dim=1)
            acc = (preds == y_test).float().mean().item()

        print(f"Epoch {epoch}/{epochs} — loss: {epoch_loss/steps_per_epoch:.4f}  test_acc: {acc:.4f}")

    elapsed = time.time() - start_time
    print(f"Finished (torch) — time: {elapsed:.2f}s  test accuracy: {acc:.4f}")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (torch) or max_iter (sklearn)")
    parser.add_argument("--use-torch", action="store_true", help="Force using PyTorch and fail if unavailable")
    args = parser.parse_args(argv)

    X, y = load_digits(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Try to use PyTorch if available
    if args.use_torch:
        try:
            import torch  # type: ignore
        except Exception as e:
            print("PyTorch requested but not available:", e)
            sys.exit(1)

    try:
        import torch  # type: ignore
        has_torch = True
    except Exception:
        has_torch = False

    if has_torch:
        # prefer MPS (Apple Metal) if available, otherwise CPU
        import torch as _torch  # type: ignore
        device = "cpu"
        try:
            if _torch.backends.mps.is_available():
                device = "mps"
            elif _torch.cuda.is_available():
                device = "cuda"
        except Exception:
            # some builds may not have mps attribute
            device = "cpu"

        run_torch(X_train, X_test, y_train, y_test, epochs=args.epochs, device=device)
    else:
        print("PyTorch not available — falling back to scikit-learn implementation")
        run_sklearn(X_train, X_test, y_train, y_test, epochs=args.epochs)


if __name__ == "__main__":
    main()
