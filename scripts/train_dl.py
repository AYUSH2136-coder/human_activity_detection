"""
Script to train, evaluate, and save PyTorch Deep Learning models.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import copy

from src.models import MLPClassifier, CNN1DClassifier, LSTMClassifier, GRUClassifier
from src.evaluation import calculate_metrics, plot_confusion_matrix, plot_roc_curve, plot_training_curves
from src.data.loader import get_activity_names
from src.utils.helpers import load_yaml, get_project_root, resolve_path, ensure_dirs
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_device(device_setting: str) -> torch.device:
    """Resolve PyTorch device based on configuration."""
    if device_setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_setting)


def get_dl_model(name: str, training_cfg: dict, num_features: int, num_classes: int, window_size: int) -> nn.Module:
    """Instantiate a PyTorch DL model based on name and config."""
    name = name.lower()
    if name == "mlp":
        cfg = training_cfg.get("mlp", {})
        return MLPClassifier(
            input_dim=num_features * window_size,
            num_classes=num_classes,
            hidden_layers=cfg.get("hidden_layers", [256, 128, 64]),
            dropout=cfg.get("dropout", 0.3),
            activation=cfg.get("activation", "relu"),
        )
    elif name == "cnn1d":
        cfg = training_cfg.get("cnn1d", {})
        return CNN1DClassifier(
            num_features=num_features,
            num_classes=num_classes,
            num_filters=cfg.get("num_filters", [64, 128, 256]),
            kernel_size=cfg.get("kernel_size", 3),
            dropout=cfg.get("dropout", 0.3),
        )
    elif name == "lstm":
        cfg = training_cfg.get("lstm", {})
        return LSTMClassifier(
            num_features=num_features,
            num_classes=num_classes,
            hidden_size=cfg.get("hidden_size", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.3),
            bidirectional=cfg.get("bidirectional", False),
        )
    elif name == "gru":
        cfg = training_cfg.get("gru", {})
        return GRUClassifier(
            num_features=num_features,
            num_classes=num_classes,
            hidden_size=cfg.get("hidden_size", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.3),
            bidirectional=cfg.get("bidirectional", False),
        )
    else:
        raise ValueError(f"Unknown Deep Learning model name: {name}")


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch Deep Learning Models")
    parser.add_argument("--paths-config", type=str, default="configs/paths.yaml", help="Path to paths config file")
    parser.add_argument("--train-config", type=str, default="configs/training.yaml", help="Path to training config file")
    args = parser.parse_args()

    root = get_project_root()
    paths_cfg = load_yaml(resolve_path(args.paths_config, root))
    train_cfg = load_yaml(resolve_path(args.train_config, root))
    
    # Seeding
    seed = train_cfg.get("general", {}).get("random_seed", 42)
    set_seed(seed)
    
    # Resolve device
    device_setting = train_cfg.get("general", {}).get("device", "auto")
    device = get_device(device_setting)
    logger.info(f"Using device: {device}")
    
    # Ensure folders exist
    ensure_dirs(
        resolve_path(paths_cfg["models"]["dl_dir"], root),
        resolve_path(paths_cfg["outputs"]["figures_dir"], root)
    )
    
    # Load dataset
    logger.info("Loading training and testing windows...")
    X_train = np.load(resolve_path(paths_cfg["data"]["X_train"], root))
    X_test = np.load(resolve_path(paths_cfg["data"]["X_test"], root))
    y_train = np.load(resolve_path(paths_cfg["data"]["y_train"], root))
    y_test = np.load(resolve_path(paths_cfg["data"]["y_test"], root))
    
    num_classes = len(np.unique(y_train))
    num_features = X_train.shape[2]
    window_size = X_train.shape[1]
    
    # Split training data into train and validation for early stopping
    # Let's use a 10% validation split
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=seed, stratify=y_train
    )
    
    activity_names = get_activity_names()
    
    # Create PyTorch datasets and loaders
    dl_cfg = train_cfg.get("dataloader", {})
    batch_size = dl_cfg.get("batch_size", 256)
    num_workers = train_cfg.get("general", {}).get("num_workers", 0)
    pin_memory = dl_cfg.get("pin_memory", True) and (device.type == "cuda")
    
    train_dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    dl_models = ["mlp", "cnn1d", "lstm", "gru"]
    results = []
    
    for name in dl_models:
        logger.info(f"\n==================== Training DL Model: {name.upper()} ====================")
        model = get_dl_model(name, train_cfg, num_features, num_classes, window_size)
        model.to(device)
        
        cfg = train_cfg.get(name, {})
        epochs = cfg.get("epochs", 50)
        lr = cfg.get("learning_rate", 0.001)
        weight_decay = cfg.get("weight_decay", 0.0001)
        patience = cfg.get("patience", 10)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Scheduler
        sch_cfg = train_cfg.get("scheduler", {})
        sch_type = sch_cfg.get("type", "none").lower()
        if sch_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sch_cfg.get("T_max", epochs))
        elif sch_type == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sch_cfg.get("step_size", 10), gamma=0.1)
        else:
            scheduler = None
            
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
        
        best_val_loss = float("inf")
        best_model_wts = copy.deepcopy(model.state_dict())
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += torch.sum(preds == targets).item()
                train_total += targets.size(0)
                
            if scheduler is not None:
                scheduler.step()
                
            epoch_train_loss = train_loss / train_total
            epoch_train_acc = train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += torch.sum(preds == targets).item()
                    val_total += targets.size(0)
                    
            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            
            # Record history
            history["train_loss"].append(epoch_train_loss)
            history["val_loss"].append(epoch_val_loss)
            history["train_acc"].append(epoch_train_acc)
            history["val_acc"].append(epoch_val_acc)
            
            logger.info(
                f"Epoch {epoch:2d}/{epochs} | "
                f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f}"
            )
            
            # Early Stopping
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break
                    
        # Load best weights
        model.load_state_dict(best_model_wts)
        
        # Test Evaluation
        logger.info(f"Evaluating {name.upper()} on test set...")
        test_correct = 0
        test_total = 0
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        # Compute metrics
        metrics = calculate_metrics(y_test, all_preds, activity_names)
        metrics["Model"] = name
        results.append(metrics)
        
        # Save model checkpoint
        dl_dir = resolve_path(paths_cfg["models"]["dl_dir"], root)
        model_path = dl_dir / f"{name}_model.pth"
        torch.save(best_model_wts, model_path)
        logger.info(f"Saved {name} weights to {model_path}")
        
        # Generate figures
        fig_dir = resolve_path(paths_cfg["outputs"]["figures_dir"], root)
        
        # Training Curves
        plot_training_curves(
            history=history,
            save_path=fig_dir / f"training_curves_{name}.png",
            title=f"Training History - {name.upper()}"
        )
        
        # Confusion Matrix
        plot_confusion_matrix(
            y_true=y_test,
            y_pred=all_preds,
            activity_names=activity_names,
            save_path=fig_dir / f"confusion_matrix_{name}.png",
            title=f"Confusion Matrix - {name.upper()}"
        )
        
        # ROC Curve
        plot_roc_curve(
            y_true=y_test,
            y_score=all_probs,
            activity_names=activity_names,
            save_path=fig_dir / f"roc_{name}.png",
            title=f"ROC Curve - {name.upper()}"
        )
        
    # Save comparison table
    df_results = pd.DataFrame(results)
    results_csv = resolve_path(paths_cfg["outputs"]["dl_results"], root)
    df_results.to_csv(results_csv, index=False)
    logger.info(f"\nAll DL training finished. Saved results comparison to {results_csv}")
    print(df_results[["Model", "accuracy", "f1_macro", "f1_weighted"]])


if __name__ == "__main__":
    main()
