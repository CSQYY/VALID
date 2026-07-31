import argparse
import os
import numpy as np
import time
import random
import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR

from dataloader import DataLoader
from timesformer.models.vit import TimeSformer

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_options():
    parser = argparse.ArgumentParser(description="Deception Detection with Multi-Task Hybrid Loss")
    parser.add_argument('--device_ID', type=str, default="cuda:0")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=51)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--len', type=int, default=64)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=15)
    # Multi-task loss weights
    parser.add_argument('--alpha_ce', type=float, default=1.0, help='Weight for main CE loss')
    parser.add_argument('--beta_aux', type=float, default=0.15, help='Max weight for AU/Visual aux losses')
    parser.add_argument('--gamma_aed', type=float, default=0.1, help='Max weight for AED alignment loss')
    parser.add_argument('--lambda_reg', type=float, default=0.1, help='Weight for AED confidence regularization')
    parser.add_argument('--ramp_epochs', type=int, default=15, help='Epochs to ramp up aux/AED losses')
    opts = parser.parse_args()
    opts.device = torch.device(opts.device_ID)
    return opts


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, initial_lr=1e-4, eta_min=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(eta_min / initial_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]


def get_class_weights(loader, num_classes=2):
    """Compute inverse-frequency class weights from training loader."""
    labels = []
    for batch in loader:
        labels.extend(batch[2].cpu().numpy())
    labels = np.array(labels, dtype=int)
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = torch.tensor([total / (num_classes * max(c, 1)) for c in counts], dtype=torch.float32)
    return weights


def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs > threshold).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    try:
        auc_val = roc_auc_score(labels, probs)
    except ValueError:
        auc_val = 0.5
    return acc, f1, auc_val


def find_optimal_threshold(labels, probs):
    thresholds = np.arange(0.25, 0.75, 0.01)
    best_f1, best_thr = 0.0, 0.5
    for thr in thresholds:
        _, f1, _ = compute_metrics(labels, probs, thr)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


class HybridDeceptionLoss(nn.Module):
    """
    Multi-task Hybrid Deception Loss:
    L = alpha * CE_main
      + beta * (CE_au + CE_vid) / 2
      + gamma * (L_align + lambda_reg * L_reg)

    beta and gamma are linearly ramped up from 0 to max over ramp_epochs.
    """

    def __init__(self, alpha=1.0, beta_max=0.15, gamma_max=0.1, lambda_reg=0.1,
                 ramp_epochs=15, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.beta_max = beta_max
        self.gamma_max = gamma_max
        self.lambda_reg = lambda_reg
        self.ramp_epochs = ramp_epochs
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits_main, event_set, logits_au, logits_vid, labels, epoch):
        # 1. Main classification loss
        loss_ce_main = self.ce(logits_main, labels)

        # 2. Auxiliary branch losses (AU + Visual)
        loss_ce_au = self.ce(logits_au, labels)
        loss_ce_vid = self.ce(logits_vid, labels)
        loss_aux = (loss_ce_au + loss_ce_vid) / 2.0

        # 3. AED confidence alignment: match AED event confidence with main classifier confidence
        aed_conf = event_set['confidence'].mean(dim=1)  # (B,)
        main_conf = torch.max(F.softmax(logits_main, dim=1), dim=1)[0]  # (B,)
        loss_conf_align = F.mse_loss(aed_conf, main_conf)

        # 4. AED confidence regularization: encourage high confidence for detected events
        loss_conf_reg = -aed_conf.mean()

        # Dynamic weight ramp-up for auxiliary and AED losses
        ratio = min(epoch / max(self.ramp_epochs, 1), 1.0)
        beta = self.beta_max * ratio
        gamma = self.gamma_max * ratio

        total_loss = (self.alpha * loss_ce_main +
                      beta * loss_aux +
                      gamma * (loss_conf_align + self.lambda_reg * loss_conf_reg))

        return total_loss


def run_epoch(args, loader, model, loss_fn, optimizer=None, scaler=None, epoch_idx=0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    losses, probs_list, labels_list = [], [], []
    desc = f"{'Train' if is_train else 'Val'} | E{epoch_idx + 1}"
    pbar = tqdm(loader, desc=desc, leave=False, unit="batch", dynamic_ncols=True)

    with torch.set_grad_enabled(is_train):
        for batch in pbar:
            # DataLoader returns (img, AUs, label, video_name) without SP
            img, AUs, label = batch[0], batch[1], batch[2]
            img = img.to(args.device, non_blocking=True)
            AUs = AUs.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            with autocast(enabled=True):
                # Model returns 4-tuple: (logits_main, event_set, logits_au, logits_vid)
                logits_main, event_set, logits_au, logits_vid = model((img, AUs))
                loss = loss_fn(logits_main, event_set, logits_au, logits_vid,
                               label, epoch=epoch_idx)

            if is_train and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            loss_val = loss.item() if torch.isfinite(loss) else np.nan
            losses.append(loss_val)

            prob = F.softmax(logits_main, dim=1)[:, 1]
            probs_list.append(prob.detach().cpu().numpy())
            labels_list.append(label.detach().cpu().numpy())
            pbar.set_postfix({'loss': f"{loss_val:.4f}" if not np.isnan(loss_val) else "NaN"})

    valid_losses = [l for l in losses if not np.isnan(l)]
    if not valid_losses:
        return 0.0, 0.0, 0.0, 0.5, 0.5

    labels_flat = np.concatenate(labels_list).flatten().astype(int)
    probs_flat = np.concatenate(probs_list).flatten()
    avg_loss = np.mean(valid_losses)

    if is_train:
        acc, f1, auc_val = compute_metrics(labels_flat, probs_flat, threshold=0.5)
        opt_thr = 0.5
    else:
        opt_thr, f1 = find_optimal_threshold(labels_flat, probs_flat)
        acc, _, auc_val = compute_metrics(labels_flat, probs_flat, opt_thr)

    return avg_loss, acc, f1, auc_val, opt_thr


def train_test(train_sets, test_sets, args):
    set_random_seed(args.seed)
    root_folder = "/data01/behavior_group/d21_qiaoyy/DOLOs/image/"

    empt_file = [
        'LS_WILTY_EP13_lie12', 'LS_WILTY_EP73_ lie1', 'LS_WILTY_EP8_lie12', 'LS_WILTY_EP8_lie14',
        'LS_WILTY_EP9_lie2', 'LS_WILTY_EP9_lie3', 'SJ_WILTY_EP71_true_12', 'SJ_WILTY_EP71_true_14',
        'SJ_WILTY_EP71_true_16', 'SJ_WILTY_EP71_true_31', 'YW_WILTY_EP47_lie19', 'YW_WILTY_EP51_lie1',
        'YW_WILTY_EP52_lie2', 'YW_WILTY_EP52_lie4', 'YW_WILTY_EP52_lie6', 'YW_WILTY_EP43_truth1'
    ]
    train_set = [d for d in train_sets if d not in empt_file]
    test_set = [d for d in test_sets if d not in empt_file]
    print(f"Data: Train={len(train_set)}, Val={len(test_set)}")

    # DataLoader no longer loads SP
    train_dataset = DataLoader(video_folder=root_folder, img_size=args.size, frame_len=args.len, video_names=train_set)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=4, drop_last=True, pin_memory=True)

    test_dataset = DataLoader(video_folder=root_folder, img_size=args.size, frame_len=args.len, video_names=test_set)
    val_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=4, drop_last=False, pin_memory=True)

    model = TimeSformer(img_size=args.size, num_classes=2, num_frames=args.len,
                        attention_type='divided_space_time').to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                  weight_decay=args.weight_decay)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=args.warmup_epochs,
                                            total_epochs=args.num_epochs,
                                            initial_lr=args.lr, eta_min=1e-6)

    # Compute class weights for imbalanced deception detection
    class_weights = get_class_weights(train_loader, num_classes=2).to(args.device)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    loss_fn = HybridDeceptionLoss(
        alpha=args.alpha_ce,
        beta_max=args.beta_aux,
        gamma_max=args.gamma_aed,
        lambda_reg=args.lambda_reg,
        ramp_epochs=args.ramp_epochs,
        class_weights=class_weights
    ).to(args.device)

    scaler = GradScaler(enabled=True)

    best_f1, best_epoch, no_improve = 0.0, -1, 0
    best_metrics = {'acc': 0.0, 'f1': 0.0, 'auc': 0.0, 'thr': 0.5}
    os.makedirs('checkpoints', exist_ok=True)

    print(f"\n{'=' * 110}")
    print(f"{'Epoch':^5} | {'Time':^6} | {'Train Loss':^10} {'Train Acc':^9} {'Train F1':^9} {'Train AUC':^9} | "
          f"{'Val Loss':^10} {'Val Acc':^9} {'Val F1':^9} {'Val AUC':^9}")
    print(f"{'=' * 110}")

    for epoch in range(args.num_epochs):
        t_start = time.time()
        t_loss, t_acc, t_f1, t_auc, _ = run_epoch(
            args, train_loader, model, loss_fn, optimizer, scaler, epoch)
        t_time = time.time() - t_start

        v_loss, v_acc, v_f1, v_auc, v_thr = run_epoch(
            args, val_loader, model, loss_fn, None, None, epoch)

        print(f"Epoch {epoch + 1:02d} | {t_time:5.1f}s | {t_loss:10.4f} {t_acc:9.4f} {t_f1:9.4f} {t_auc:9.4f} | "
              f"{v_loss:10.4f} {v_acc:9.4f} {v_f1:9.4f} {v_auc:9.4f}")

        scheduler.step()

        if not np.isnan(v_f1) and v_f1 > best_f1:
            best_f1, best_epoch, no_improve = v_f1, epoch, 0
            best_metrics = {'acc': v_acc, 'f1': v_f1, 'auc': v_auc, 'thr': v_thr}
            ckpt = f"checkpoints/best_e{epoch + 1}_f1{v_f1:.4f}.pth"
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'metrics': best_metrics,
                'class_weights': class_weights.cpu(),
            }, ckpt)
            print(f"   New Best -> Saved: {ckpt}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n Early Stopping at Epoch {epoch + 1}. Best Val F1: {best_f1:.4f} @ Epoch {best_epoch + 1}")
                break

    print(f"{'=' * 110}")
    print(f"Finished! Best Val F1: {best_metrics['f1']:.4f} @ Epoch {best_epoch + 1} (Thr: {best_metrics['thr']:.2f})")
    return (f"Best Acc: {best_metrics['acc']:.4f}, F1: {best_metrics['f1']:.4f}, "
            f"AUC: {best_metrics['auc']:.4f}")


if __name__ == "__main__":
    opts = parse_options()
    root_path = '/data01/behavior_group/d21_qiaoyy/DOLOs/protocols/'
    train_roots = ['train_fold1.csv', 'train_fold2.csv', 'train_fold3.csv']
    test_roots = ['test_fold1.csv', 'test_fold2.csv', 'test_fold3.csv']

    os.makedirs('Outputs', exist_ok=True)
    with open('Outputs/results.txt', 'w') as f:
        for i in range(len(test_roots)):
            print(f"\n{'#' * 25} Fold {i + 1}/{len(test_roots)} {'#' * 25}")
            train_fs = read_file(root_path + train_roots[i])
            test_fs = read_file(root_path + test_roots[i])
            res = train_test(train_fs, test_fs, opts)
            f.write(f"Fold {i + 1}: {res}\n")
    print("\nAll folds done. Results saved.")