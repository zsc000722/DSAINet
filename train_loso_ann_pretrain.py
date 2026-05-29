import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import inspect
import copy
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from utils.util import set_seed, setup_logger, classwise_augmentation
from utils.load_data import load_data
from model.DSAINet_SNN_ANN import DSAINet_SNN_PLIF, DSAINet_ANN_Symmetric


def parse_args():
    parser = argparse.ArgumentParser(description="ANN warm-up then PLIF-SNN LOSO training")
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=150, help='total epochs: first half ANN, second half SNN')
    parser.add_argument('--ann-epochs', type=int, default=50, help='ANN warm-up epochs; default = epochs // 2')
    parser.add_argument('--snn-lr', type=float, default=None, help='optional learning rate after switching to SNN')
    parser.add_argument('--times', type=int, default=100, help='run id used in log name')
    return parser.parse_args()


def _filtered_model_kwargs(model_cls, model_params, n_class, n_channels, n_times):
    params = dict(model_params)
    params.pop('name', None)
    sig = inspect.signature(model_cls.__init__)
    allowed = set(sig.parameters.keys()) - {'self'}
    params.update({
        'n_classes': n_class,
        'Chans': n_channels,
        'Samples': n_times,
    })
    return {k: v for k, v in params.items() if k in allowed}


def make_optimizer(model, config, lr=None):
    train_cfg = config.get('train', {})
    lr = float(train_cfg.get('lr', 1e-3) if lr is None else lr)
    wd = float(train_cfg.get('weight_decay', train_cfg.get('wd', 0.0)))
    opt_name = str(train_cfg.get('optimizer', 'Adam')).lower()
    if opt_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def make_criterion(config):
    train_cfg = config.get('train', {})
    label_smoothing = float(train_cfg.get('label_smoothing', 0.0))
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def transfer_ann_to_snn(ann_model, snn_model, logger=None):
    """Copy all shape-compatible ANN parameters/buffers into the SNN model."""
    ann_sd = ann_model.state_dict()
    snn_sd = snn_model.state_dict()
    compatible = {}
    skipped = []
    for k, v in ann_sd.items():
        if k in snn_sd and tuple(snn_sd[k].shape) == tuple(v.shape):
            compatible[k] = v
        else:
            skipped.append(k)
    snn_sd.update(compatible)
    missing, unexpected = snn_model.load_state_dict(snn_sd, strict=False)
    if logger is not None:
        logger.info(f"ANN->SNN transfer: copied {len(compatible)} tensors; skipped {len(skipped)} ANN tensors.")
        if skipped:
            logger.info(f"ANN->SNN transfer skipped examples: {skipped[:20]}")
        if missing:
            logger.info(f"ANN->SNN missing SNN tensors after load examples: {missing[:20]}")
        if unexpected:
            logger.info(f"ANN->SNN unexpected tensors examples: {unexpected[:20]}")
    return compatible, skipped


def train_one_epoch(model, loader, criterion, optimizer, device, n_segments=8):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        X, y = classwise_augmentation(X, y, n_segments=n_segments)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss / max(len(loader), 1), correct / max(total, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            outputs = model(X)
            loss = criterion(outputs, y)
            loss_sum += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_targets.extend(y.detach().cpu().numpy().tolist())
    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(len(loader), 1)
    return avg_loss, acc, all_preds, all_targets


def train_test_loso(data, labels, config, device, logger, args, n_class, n_channels, n_times):
    n_subjects = len(data)
    batch_size = config['train']['batch_size']
    num_epochs = config['train']['epochs']
    ann_epochs = args.ann_epochs if args.ann_epochs is not None else int(config['train'].get('ann_epochs', num_epochs // 2))
    ann_epochs = max(0, min(ann_epochs, num_epochs))
    snn_lr = args.snn_lr if args.snn_lr is not None else config['train'].get('snn_lr', config['train']['lr'])

    logger.info("========== Start LOSO ANN-warmup -> PLIF-SNN Training ==========")
    logger.info(f"Schedule: ANN warm-up epochs = {ann_epochs}; SNN fine-tune epochs = {num_epochs - ann_epochs}")

    all_val_subject_acc, all_val_subject_kappa, all_val_subject_f1 = [], [], []
    all_test_subject_acc, all_test_subject_kappa, all_test_subject_f1 = [], [], []

    for test_subj in range(n_subjects):
        model_params = config['model']
        train_data_list = [data[i] for i in range(n_subjects) if i != test_subj]
        train_labels_list = [labels[i] for i in range(n_subjects) if i != test_subj]

        val_data_list, val_labels_list = [], []
        new_train_data_list, new_train_labels_list = [], []
        for data_subj, labels_subj in zip(train_data_list, train_labels_list):
            X_train, X_val, y_train, y_val = train_test_split(
                data_subj,
                labels_subj,
                test_size=0.2,
                stratify=labels_subj,
                random_state=config['train']['seed']
            )
            new_train_data_list.append(X_train)
            new_train_labels_list.append(y_train)
            val_data_list.append(X_val)
            val_labels_list.append(y_val)

        train_data = np.concatenate(new_train_data_list, axis=0)
        train_labels = np.concatenate(new_train_labels_list, axis=0)
        valid_data = np.concatenate(val_data_list, axis=0)
        valid_labels = np.concatenate(val_labels_list, axis=0)
        test_data = data[test_subj]
        test_labels = labels[test_subj]

        train_data = train_data[:, None, :, :]
        valid_data = valid_data[:, None, :, :]
        test_data  = test_data[:, None, :, :]

        if config['train']['norm'] == 'Z_Score':
            train_mean = train_data.mean(axis=(0, 1, 3), keepdims=True)
            train_std  = train_data.std(axis=(0, 1, 3), keepdims=True)
            train_data = (train_data - train_mean) / train_std
            valid_data = (valid_data - train_mean) / train_std
            test_data  = (test_data - train_mean) / train_std

        ann_kwargs = _filtered_model_kwargs(DSAINet_ANN_Symmetric, model_params, n_class, n_channels, n_times)
        snn_kwargs = _filtered_model_kwargs(DSAINet_SNN_PLIF, model_params, n_class, n_channels, n_times)
        ann_model = DSAINet_ANN_Symmetric(**ann_kwargs).to(device)
        snn_model = DSAINet_SNN_PLIF(**snn_kwargs).to(device)
        criterion = make_criterion(config).to(device)
        ann_optimizer = make_optimizer(ann_model, config, lr=config['train']['lr'])
        snn_optimizer = make_optimizer(snn_model, config, lr=snn_lr)

        logger.info(f"====== Subject {test_subj + 1}/{n_subjects} as Test ======")
        logger.info(f"ANN Parameters: {sum(p.numel() for p in ann_model.parameters() if p.requires_grad)}")
        logger.info(f"SNN Parameters: {sum(p.numel() for p in snn_model.parameters() if p.requires_grad)}")
        logger.info(f"Train size = {train_data.shape[0]}, Valid size = {valid_data.shape[0]}, Test size = {test_data.shape[0]}")

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_data, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.long)
        )
        valid_dataset = torch.utils.data.TensorDataset(
            torch.tensor(valid_data, dtype=torch.float32),
            torch.tensor(valid_labels, dtype=torch.long)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(test_data, dtype=torch.float32),
            torch.tensor(test_labels, dtype=torch.long)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, persistent_workers=True,
                                  prefetch_factor=2)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=2, pin_memory=True, persistent_workers=True)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=2, pin_memory=True, persistent_workers=True)

        # ANN warm-up stage. Metrics are logged but final best checkpoints are selected from SNN stage.
        for epoch in range(ann_epochs):
            train_loss, train_acc = train_one_epoch(
                ann_model, train_loader, criterion, ann_optimizer, device,
                n_segments=config['train'].get('n_segments', 8)
            )
            val_loss, val_acc, _, _ = evaluate(ann_model, valid_loader, criterion, device)
            test_loss, test_acc, _, _ = evaluate(ann_model, test_loader, criterion, device)
            logger.info(
                f"Sub {test_subj+1} | [ANN warmup] Epoch {epoch+1}/{num_epochs} | "
                f"train Loss: {train_loss:.4f} | train Accuracy: {train_acc:.4f} | "
                f"valid Loss: {val_loss:.4f} | valid Accuracy: {val_acc:.4f} | "
                f"test Accuracy: {test_acc:.4f}"
            )

        transfer_ann_to_snn(ann_model, snn_model, logger=logger)
        del ann_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        best_val_acc = 0.0
        best_val_loss = float('inf')
        best_test_acc = 0.0
        best_test_loss = float('inf')
        best_val_model_state = copy.deepcopy(snn_model.state_dict())
        best_test_model_state = copy.deepcopy(snn_model.state_dict())
        best_val_epoch = ann_epochs
        best_test_epoch = ann_epochs
        patient = 0

        for epoch in range(ann_epochs, num_epochs):
            train_loss, train_acc = train_one_epoch(
                snn_model, train_loader, criterion, snn_optimizer, device,
                n_segments=config['train'].get('n_segments', 8)
            )
            val_loss, val_acc, _, _ = evaluate(snn_model, valid_loader, criterion, device)
            test_loss, test_acc, _, _ = evaluate(snn_model, test_loader, criterion, device)

            if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_val_model_state = copy.deepcopy(snn_model.state_dict())
                best_val_epoch = epoch + 1
                patient = 0
                logger.info(f"Sub {test_subj+1} | Early Stopping: Best SNN Epoch {epoch+1}/{num_epochs}")
            else:
                patient += 1

            if test_acc > best_test_acc or (test_acc == best_test_acc and test_loss < best_test_loss):
                best_test_acc = test_acc
                best_test_loss = test_loss
                best_test_epoch = epoch + 1
                best_test_model_state = copy.deepcopy(snn_model.state_dict())
                logger.info(f"Sub {test_subj+1} | All Epoch: Best SNN Epoch {epoch+1}/{num_epochs}")

            logger.info(
                f"Sub {test_subj+1} | [SNN finetune] Epoch {epoch+1}/{num_epochs} | "
                f"train Loss: {train_loss:.4f} | train Accuracy: {train_acc:.4f} | "
                f"valid Loss: {val_loss:.4f} | valid Accuracy: {val_acc:.4f} | "
                f"test Accuracy: {test_acc:.4f} | patient: {patient}"
            )

        save_dir = f"/mnt/data/250010236/DSAINet/weight/{config['model']['name']}/{args.dataset}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_val_model_state, f"{save_dir}/{test_subj}.pth")
        torch.save(best_test_model_state, f"{save_dir}/best_{test_subj}.pth")

        snn_model.load_state_dict(best_val_model_state)
        test_loss, acc, all_preds, all_targets = evaluate(snn_model, test_loader, criterion, device)
        kappa = cohen_kappa_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        all_val_subject_acc.append(acc)
        all_val_subject_kappa.append(kappa)
        all_val_subject_f1.append(f1)
        logger.info(f"Early Stopping Best Epoch: {best_val_epoch} | Test Subject {test_subj + 1} | Accuracy: {acc:.4f} | Kappa: {kappa:.4f} | F1: {f1:.4f}")

        snn_model.load_state_dict(best_test_model_state)
        test_loss, acc, all_preds, all_targets = evaluate(snn_model, test_loader, criterion, device)
        kappa = cohen_kappa_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        all_test_subject_acc.append(acc)
        all_test_subject_kappa.append(kappa)
        all_test_subject_f1.append(f1)
        logger.info(f"All Best Epoch: {best_test_epoch} | Test Subject {test_subj + 1} | Accuracy: {acc:.4f} | Kappa: {kappa:.4f} | F1: {f1:.4f}")

    logger.info("========== LOSO Done ==========")
    logger.info("----- Early Stopping Results -----")
    for i, (acc, kappa, f1) in enumerate(zip(all_val_subject_acc, all_val_subject_kappa, all_val_subject_f1)):
        logger.info(f"Subject {i+1:02d} | Acc = {acc:.4f} | Kappa = {kappa:.4f} | F1 = {f1:.4f}")
    logger.info("----- All Epoch Results -----")
    for i, (acc, kappa, f1) in enumerate(zip(all_test_subject_acc, all_test_subject_kappa, all_test_subject_f1)):
        logger.info(f"Subject {i+1:02d} | Acc = {acc:.4f} | Kappa = {kappa:.4f} | F1 = {f1:.4f}")
    logger.info(f"Early Stopping - Average LOSO Accuracy: {np.mean(all_val_subject_acc):.4f}±{np.std(all_val_subject_acc):.4f}")
    logger.info(f"Early Stopping - Average LOSO Kappa: {np.mean(all_val_subject_kappa):.4f}±{np.std(all_val_subject_kappa):.4f}")
    logger.info(f"Early Stopping - Average LOSO F1: {np.mean(all_val_subject_f1):.4f}±{np.std(all_val_subject_f1):.4f}")
    logger.info(f"All Epoch - Average LOSO Accuracy: {np.mean(all_test_subject_acc):.4f}±{np.std(all_test_subject_acc):.4f}")
    logger.info(f"All Epoch - Average LOSO Kappa: {np.mean(all_test_subject_kappa):.4f}±{np.std(all_test_subject_kappa):.4f}")
    logger.info(f"All Epoch - Average LOSO F1: {np.mean(all_test_subject_f1):.4f}±{np.std(all_test_subject_f1):.4f}")


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        set_seed(args.seed)
        config['train']['seed'] = args.seed
    else:
        set_seed(config['train']['seed'])
    if args.batch_size is not None:
        config['train']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['train']['epochs'] = args.epochs
    if args.lr is not None:
        config['train']['lr'] = args.lr
    if args.snn_lr is not None:
        config['train']['snn_lr'] = args.snn_lr
    if args.ann_epochs is not None:
        config['train']['ann_epochs'] = args.ann_epochs

    logger = setup_logger(
        f"{args.dataset}_{args.times}_ann_to_plif_snn",
        log_dir=f"/mnt/data/250010236/DSAINet/log/{config['model']['name']}/",
        overwrite=True,
    )
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    logger.info("==== Experiment Config ====")
    logger.info(yaml.dump(config).rstrip())
    logger.info("==========================")

    data, labels, n_class, n_channels, n_times = load_data(args.dataset, 'LOSO')
    train_test_loso(data, labels, config, device, logger, args, n_class, n_channels, n_times)
