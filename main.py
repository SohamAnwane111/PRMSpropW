#!/usr/bin/env python3
"""

Advanced experiment comparing Vanilla RMSProp vs PRMSPropW on Neural Networks
CPU-ONLY VERSION (All CUDA logic removed)
- Classification datasets (tabular)
- Regression dataset: California Housing
- Training curves and metrics saving
- NO contour / loss landscape plotting

Added:
- Cohen's kappa metric (train & validation) and plotting of kappa curves
- Training F1 score tracking and plotting
"""

import os
import time
import argparse
import warnings
from typing import List
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import (
    load_wine,
    load_breast_cancer,
    fetch_openml,
    fetch_california_housing,
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    cohen_kappa_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def now_str() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def makedirs(path: str):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------
# Dataset Loading (NON-IMAGE)
# ---------------------------------------------------------------------
def load_dataset(name: str = 'wine_quality'):
    """Load datasets including regression (California Housing)."""
    print(f"📊 Loading {name} dataset...")

    if name == 'wine_quality':
        try:
            data = fetch_openml('wine-quality-red', version=1, parser='auto')
            X = data.data.values
            y = data.target.values.astype(float)
            # Map to 3 classes
            y_classes = np.zeros_like(y, dtype=int)
            y_classes[y <= 5] = 0
            y_classes[(y > 5) & (y <= 6)] = 1
            y_classes[y > 6] = 2
            y = y_classes
            info = {
                'name': 'Wine Quality (UCI)',
                'type': 'Classification (3 classes)',
                'features': X.shape[1],
                'samples': len(X),
                'target': 'Quality class',
            }
        except Exception:
            data = load_wine()
            X, y = data.data, data.target
            info = {
                'name': 'Wine (sklearn fallback)',
                'type': 'Classification (3 classes)',
                'features': X.shape[1],
                'samples': len(X),
                'target': 'Cultivar',
            }

    elif name == 'adult_income':
        try:
            data = fetch_openml('adult', version=2, parser='auto')
            X = data.data
            y = data.target
            for col in X.select_dtypes(include=['object', 'category']).columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            X = X.values.astype(float)
            y = LabelEncoder().fit_transform(y)
            info = {
                'name': 'Adult Income',
                'type': 'Classification (binary)',
                'features': X.shape[1],
                'samples': len(X),
                'target': 'Income >50k',
            }
        except Exception:
            return load_dataset('breast_cancer')

    elif name == 'heart_disease':
        try:
            data = fetch_openml('heart-statlog', version=1, parser='auto')
            X = data.data.values.astype(float)
            y = LabelEncoder().fit_transform(data.target.values)
            info = {
                'name': 'Heart Disease',
                'type': 'Classification (binary)',
                'features': X.shape[1],
                'samples': len(X),
                'target': 'Disease',
            }
        except Exception:
            return load_dataset('breast_cancer')

    elif name == 'breast_cancer':
        data = load_breast_cancer()
        X, y = data.data, data.target
        info = {
            'name': 'Breast Cancer',
            'type': 'Classification (binary)',
            'features': X.shape[1],
            'samples': len(X),
            'target': 'Diagnosis',
        }

    elif name == 'bank_marketing':
        try:
            data = fetch_openml('bank-marketing', version=1, parser='auto')
            X = data.data
            y = data.target
            for col in X.select_dtypes(include=['object', 'category']).columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            X = X.values.astype(float)
            y = LabelEncoder().fit_transform(y)
            info = {
                'name': 'Bank Marketing',
                'type': 'Classification (binary)',
                'features': X.shape[1],
                'samples': len(X),
                'target': 'Subscribe to term deposit',
            }
        except Exception:
            return load_dataset('breast_cancer')

    elif name == 'credit_fraud':
        try:
            data = fetch_openml('creditcard', version=1, parser='auto')
            X = data.data.values.astype(float)
            y = data.target.values.astype(int)
            # Subsample for speed
            if len(X) > 10000:
                idx = np.random.choice(len(X), 10000, replace=False)
                X, y = X[idx], y[idx]
            info = {
                'name': 'Credit Card Fraud',
                'type': 'Binary classification (imbalanced)',
                'features': X.shape[1],
                'samples': len(X),
                'target': 'Fraud/Legit',
            }
        except Exception:
            return load_dataset('breast_cancer')

    elif name == 'california_housing':
        data = fetch_california_housing()
        X, y = data.data, data.target  # continuous
        info = {
            'name': 'California Housing (Regression)',
            'type': 'Regression (continuous)',
            'features': X.shape[1],
            'samples': len(X),
            'target': 'MedianHouseValue',
        }

    else:
        raise ValueError(f"Unknown dataset: {name}")

    print(f"✅ Loaded: {info['name']} ({info['type']})")
    return X, y, info


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class MultiLayerNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], output_dim: int = 2, dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_flat_params(self, flat: torch.Tensor):
        i = 0
        for p in self.parameters():
            n = p.numel()
            p.data = flat[i:i+n].view(p.shape)
            i += n


# ---------------------------------------------------------------------
# Custom Optimizers (CPU)
# ---------------------------------------------------------------------
class RMSPropVanilla:
    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.state = [torch.zeros_like(p.data) for p in self.params]

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad.data
                s = self.state[i]
                s.mul_(self.beta).add_(g**2, alpha=1 - self.beta)
                p.data.add_(g / (torch.sqrt(s) + self.eps), alpha=-self.lr)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


class PRMSPropW:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.vel = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad.data
                self.v[i].mul_(self.beta2).add_(g**2, alpha=1 - self.beta2)
                self.vel[i].mul_(self.beta1).add_(g)
                p.data.add_(self.vel[i] / (torch.sqrt(self.v[i]) + self.eps), alpha=-self.lr)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# ---------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------
def train_model(model, optimizer, train_loader, val_loader, criterion, epochs=50, desc="", is_classification=True):
    """Train model and collect metrics. Works for both classification and regression."""
    metrics = {
        'epoch': [], 'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_precision': [], 'val_recall': [],
        'train_kappa': [], 'val_kappa': [], 'train_f1': [],
        'grad_norm': [], 'update_norm': [], 'param_norm': []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        grad_norms = []
        prev_params = model.get_flat_params().clone()

        all_preds_train = []
        all_targets_train = []

        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()

            # gradient norm
            grad_norm = torch.sqrt(sum(
                torch.sum(p.grad ** 2) for p in model.parameters() if p.grad is not None
            ))
            grad_norms.append(grad_norm.item())

            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            train_total += xb.size(0)
            if is_classification:
                preds = out.argmax(1)
                train_correct += preds.eq(yb).sum().item()
                all_preds_train.extend(preds.cpu().numpy().tolist())
                all_targets_train.extend(yb.cpu().numpy().tolist())

        curr_params = model.get_flat_params()
        update_norm = torch.norm(curr_params - prev_params).item()
        param_norm = torch.norm(curr_params).item()

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                val_total += xb.size(0)
                if is_classification:
                    preds = out.argmax(1)
                    val_correct += preds.eq(yb).sum().item()
                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_targets.extend(yb.cpu().numpy().tolist())

        # Aggregate
        train_loss /= train_total
        if is_classification:
            train_acc = 100.0 * train_correct / train_total
        else:
            train_acc = float('nan')

        val_loss /= val_total
        if is_classification:
            val_acc = 100.0 * val_correct / val_total
            val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
            val_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
            val_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)

            # Train F1 score
            try:
                if len(all_targets_train) > 0 and len(set(all_targets_train)) >= 2:
                    train_f1 = f1_score(all_targets_train, all_preds_train, average='macro', zero_division=0)
                else:
                    train_f1 = float('nan')
            except Exception:
                train_f1 = float('nan')

            # Cohen's kappa
            try:
                if len(all_targets) > 0 and len(set(all_targets)) >= 2:
                    val_kappa = cohen_kappa_score(all_targets, all_preds)
                else:
                    val_kappa = float('nan')
            except Exception:
                val_kappa = float('nan')

            # Train kappa (from collected training preds)
            try:
                if len(all_targets_train) > 0 and len(set(all_targets_train)) >= 2:
                    train_kappa = cohen_kappa_score(all_targets_train, all_preds_train)
                else:
                    train_kappa = float('nan')
            except Exception:
                train_kappa = float('nan')
        else:
            val_acc = float('nan')
            val_f1 = float('nan')
            val_precision = float('nan')
            val_recall = float('nan')
            val_kappa = float('nan')
            train_kappa = float('nan')
            train_f1 = float('nan')

        # Store
        metrics['epoch'].append(epoch)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        metrics['val_f1'].append(val_f1)
        metrics['val_precision'].append(val_precision)
        metrics['val_recall'].append(val_recall)
        metrics['train_kappa'].append(train_kappa)
        metrics['val_kappa'].append(val_kappa)
        metrics['train_f1'].append(train_f1)
        metrics['grad_norm'].append(np.mean(grad_norms))
        metrics['update_norm'].append(update_norm)
        metrics['param_norm'].append(param_norm)

        # Log
        if is_classification:
            print(f"[{desc}] Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                  f"F1: {val_f1:.3f} | Val Kappa: {val_kappa:.3f}")
        else:
            print(f"[{desc}] Epoch {epoch:3d}/{epochs} | "
                  f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

    return metrics


# ---------------------------------------------------------------------
# Plotting: Training Curves
# ---------------------------------------------------------------------
def plot_training_curves(metrics_a, metrics_b, out_dir, title_suffix="", is_classification=True):
    """
    Plots training and validation curves for two optimizers:
    RMSProp and PRMSPropW.
    """
    makedirs(out_dir)
    epochs = metrics_a['epoch']
    ts = now_str()

    def _plot_two(ykey, ylabel, fname):
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, metrics_a[ykey], label='RMSProp', linewidth=2)
        plt.plot(epochs, metrics_b[ykey], label='PRMSPropW', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f'{ylabel} vs Epochs {title_suffix}', fontsize=13, fontweight='bold')
        plt.grid(alpha=0.3, linestyle='--')
        plt.legend(fontsize=10)
        out = os.path.join(out_dir, f"{fname}_{ts}.png")
        plt.savefig(out, dpi=160, bbox_inches='tight')
        plt.close()
        print(f"🖼️ Saved: {out}")

    # Always plot losses and norms
    _plot_two('train_loss', 'Train Loss', 'train_loss')
    _plot_two('val_loss', 'Validation Loss', 'val_loss')
    _plot_two('grad_norm', 'Gradient Norm', 'grad_norm')
    _plot_two('update_norm', 'Update Norm', 'update_norm')
    _plot_two('param_norm', 'Parameter Norm', 'param_norm')

    # Only for classification
    if is_classification:
        _plot_two('train_acc', 'Train Accuracy (%)', 'train_acc')
        _plot_two('val_acc', 'Validation Accuracy (%)', 'val_acc')
        _plot_two('train_f1', 'Train F1 (macro)', 'train_f1')
        _plot_two('val_f1', 'Validation F1 (macro)', 'val_f1')
        # Kappa plots (train & val)
        _plot_two('train_kappa', "Train Cohen's Kappa", 'train_kappa')
        _plot_two('val_kappa', "Validation Cohen's Kappa", 'val_kappa')

    print("All training curves plotted successfully.")


def save_metrics_csv(metrics, out_path):
    df = pd.DataFrame(metrics)
    df.to_csv(out_path, index=False)
    print(f"Saved metrics: {out_path}")


# ---------------------------------------------------------------------
# Evaluate on Test Set
# ---------------------------------------------------------------------
def evaluate(model, loader, criterion, is_classification=True):
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * xb.size(0)
            total += xb.size(0)

            if is_classification:
                probs = F.softmax(out, dim=1)
                pred = probs.argmax(dim=1)
                correct += (pred == yb).sum().item()
                all_preds.extend(pred.cpu().numpy().tolist())
                all_targets.extend(yb.cpu().numpy().tolist())
                all_probs.extend(
                    probs[:, 1].cpu().numpy().tolist() if probs.shape[1] > 1 else probs.squeeze().cpu().numpy().tolist()
                )

    avg_loss = total_loss / total
    if is_classification:
        acc = 100.0 * correct / total
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        try:
            if len(set(all_targets)) == 2:
                auc = roc_auc_score(all_targets, all_probs)
            else:
                auc = np.nan
        except Exception:
            auc = np.nan
        try:
            if len(all_targets) > 0 and len(set(all_targets)) >= 2:
                kappa = cohen_kappa_score(all_targets, all_preds)
            else:
                kappa = np.nan
        except Exception:
            kappa = np.nan

        return {'loss': avg_loss, 'acc': acc, 'f1': f1, 'precision': prec, 'recall': rec, 'auc': auc, 'kappa': kappa}
    else:
        return {'loss': avg_loss}


# ---------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------
def prepare_loaders(X, y, batch_size=128, test_size=0.2, val_size=0.2, stratify=True):
    # For regression, disable stratify
    is_classification = (np.array_equal(y, y.astype(int)) and len(np.unique(y)) >= 2)
    strat = y if (stratify and is_classification) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=strat
    )
    strat2 = y_train if (stratify and is_classification) else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=SEED, stratify=strat2
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)

    if is_classification:
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        y_val_t   = torch.tensor(y_val, dtype=torch.long)
        y_test_t  = torch.tensor(y_test, dtype=torch.long)
    else:
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_val_t   = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        y_test_t  = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, is_classification


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------
def run_experiment(args):
    X, y, info = load_dataset(args.dataset)

    out_root = os.path.join('outputs_prmsprop_ann', f"{args.dataset}_{now_str()}")
    makedirs(out_root)

    train_loader, val_loader, test_loader, is_cls = prepare_loaders(
        X, y, batch_size=args.batch_size, test_size=args.test_size, val_size=args.val_size, stratify=True
    )

    input_dim = X.shape[1]
    output_dim = (max(2, len(np.unique(y))) if is_cls else 1)

    # Hidden dims
    hidden_dims = [int(h.strip()) for h in args.hidden_dims.split(',')] if isinstance(args.hidden_dims, str) else args.hidden_dims

    # Build two identical models with the same init
    torch.manual_seed(SEED)
    base_model = MultiLayerNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, dropout=args.dropout)
    model_a = MultiLayerNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, dropout=args.dropout)
    model_b = MultiLayerNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, dropout=args.dropout)
    model_a.load_state_dict(base_model.state_dict())
    model_b.load_state_dict(base_model.state_dict())

    # Criterion
    if is_cls:
        # Optionally enable class weights for imbalanced binary tasks
        if args.class_weighted:
            y_counts = np.bincount(y.astype(int))
            if len(y_counts) >= 2 and np.min(y_counts) > 0:
                weights = torch.tensor((y_counts.sum() / (len(y_counts) * y_counts)).astype(np.float32))
                criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    # Optimizers
    opt_a = RMSPropVanilla(model_a.parameters(), lr=args.lr, beta=args.beta, eps=args.eps)
    opt_b = PRMSPropW(model_b.parameters(), lr=args.lr, beta1=args.beta1, beta2=args.beta2, eps=args.eps)

    # Train
    print("\n===== Training with RMSProp (Vanilla) =====")
    metrics_a = train_model(model_a, opt_a, train_loader, val_loader, criterion, epochs=args.epochs, desc="RMSProp", is_classification=is_cls)

    print("\n===== Training with PRMSPropW =====")
    metrics_b = train_model(model_b, opt_b, train_loader, val_loader, criterion, epochs=args.epochs, desc="PRMSPropW", is_classification=is_cls)

    # Save metrics & curves
    save_metrics_csv(metrics_a, os.path.join(out_root, "metrics_rmsprop.csv"))
    save_metrics_csv(metrics_b, os.path.join(out_root, "metrics_prmspropw.csv"))
    plot_training_curves(metrics_a, metrics_b, out_root, title_suffix=f"on {args.dataset}", is_classification=is_cls)
    
    # Final evaluation on test
    res_a = evaluate(model_a, test_loader, criterion, is_classification=is_cls)
    res_b = evaluate(model_b, test_loader, criterion, is_classification=is_cls)

    # Save final results
    with open(os.path.join(out_root, "final_test_results.txt"), "w") as f:
        f.write("RMSProp (Vanilla) Test Results\n")
        for k, v in res_a.items():
            f.write(f"{k}: {v}\n")
        f.write("\nPRMSPropW Test Results\n")
        for k, v in res_b.items():
            f.write(f"{k}: {v}\n")

    print("Experiment complete.")
    print("RMSProp Test:", res_a)
    print("PRMSPropW Test:", res_b)
    print(f"Output directory: {out_root}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RMSProp vs PRMSPropW on tabular NN (CPU-only; no contour plots)")
    parser.add_argument('--dataset', type=str, default='wine_quality',
                        choices=['wine_quality', 'adult_income', 'heart_disease',
                                 'bank_marketing', 'credit_fraud', 'breast_cancer',
                                 'california_housing'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--hidden_dims', type=str, default='128,64')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=0.9, help='RMSProp beta')
    parser.add_argument('--beta1', type=float, default=0.9, help='PRMSPropW momentum')
    parser.add_argument('--beta2', type=float, default=0.99, help='PRMSPropW RMSProp beta')
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--class_weighted', action='store_true', help='Use class weights for CE loss (classification only)')
    args = parser.parse_args()

    run_experiment(args)


if __name__ == '__main__':
    main()