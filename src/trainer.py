import csv
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, confusion_matrix)

LR      = 0.0001
EPOCHS  = 100
CLASSES = ['Normal', 'Tumor']

_MEAN = np.array([0.485, 0.456, 0.406])
_STD  = np.array([0.229, 0.224, 0.225])


# ── Utilitários ───────────────────────────────────────────────────────────────

def _gpu_mem():
    return f'{torch.cuda.memory_reserved()/1e9:.1f}G' if torch.cuda.is_available() else 'CPU'


def _denorm(t):
    img = t.cpu().numpy().transpose(1, 2, 0) * _STD + _MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _save_batch_grid(imgs, labels, path, title, preds=None):
    n    = min(len(imgs), 16)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()
    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            ax.imshow(_denorm(imgs[i]))
            true_lbl = CLASSES[int(labels[i])]
            if preds is not None:
                pred_lbl = CLASSES[int(preds[i])]
                color = 'green' if pred_lbl == true_lbl else 'red'
                ax.set_title(f'T:{true_lbl}\nP:{pred_lbl}', fontsize=7, color=color)
            else:
                ax.set_title(true_lbl, fontsize=8)
        ax.axis('off')
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()


# ── Loop de treino ────────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, criterion, device,
                 epoch, total_epochs, run_dir):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    bar = tqdm(loader, desc=f'  {epoch:3d}/{total_epochs} [treino]',
               leave=False, ncols=110, unit='batch')

    for n, (imgs, labels) in enumerate(bar, 1):
        if epoch == 1 and n <= 3:
            _save_batch_grid(imgs, labels,
                             run_dir / f'train_batch{n-1}.jpg',
                             f'Train Batch {n-1}')

        imgs   = imgs.to(device)
        labels = labels.to(device, dtype=torch.long)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds   = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        bar.set_postfix(loss=f'{total_loss/n:.4f}', acc=f'{correct/total*100:.2f}%')

    return total_loss / len(loader), correct / total


def _evaluate(model, loader, criterion, device, desc,
              run_dir=None, batch_prefix=None):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, batch_imgs, batch_lbls = [], [], [], []
    bar = tqdm(loader, desc=desc, leave=False, ncols=110, unit='batch')

    with torch.no_grad():
        for n, (imgs, labels) in enumerate(bar, 1):
            out  = model(imgs.to(device))
            loss = criterion(out, labels.to(device, dtype=torch.long))
            preds = out.argmax(dim=1).cpu()

            total_loss += loss.item()
            correct    += (preds == labels.long()).sum().item()
            total      += labels.size(0)
            all_preds.extend(preds.numpy().astype(int))
            all_labels.extend(labels.numpy().astype(int))

            if run_dir and batch_prefix and n <= 3:
                _save_batch_grid(imgs, labels,
                                 run_dir / f'{batch_prefix}{n-1}_labels.jpg',
                                 f'Val Batch {n-1} — Labels')
                _save_batch_grid(imgs, labels,
                                 run_dir / f'{batch_prefix}{n-1}_pred.jpg',
                                 f'Val Batch {n-1} — Predictions',
                                 preds=preds)

            bar.set_postfix(loss=f'{total_loss/n:.4f}',
                            acc=f'{correct/total*100:.2f}%')

    return (total_loss / len(loader), correct / total,
            np.array(all_labels), np.array(all_preds))


# ── Plots finais ──────────────────────────────────────────────────────────────

def _save_results_png(history, run_dir):
    ep = [r['epoch'] for r in history]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    pairs = [
        (axes[0,0], 'train/loss',        'val/loss',        'Loss'),
        (axes[0,1], 'train/acc',         'val/acc',         'Accuracy (%)'),
        (axes[0,2], 'metrics/precision', None,              'Precision (%)'),
        (axes[1,0], 'metrics/recall',    None,              'Recall (%)'),
        (axes[1,1], 'metrics/f1',        None,              'F1 (%)'),
    ]
    for ax, tr_key, vl_key, title in pairs:
        ax.plot(ep, [r[tr_key] for r in history], label='train' if vl_key else title)
        if vl_key:
            ax.plot(ep, [r[vl_key] for r in history], label='val')
            ax.legend(fontsize=8)
        ax.set_title(title)
        ax.set_xlabel('Época')
        ax.grid(True, alpha=0.3)

    axes[1,2].axis('off')
    plt.suptitle('Training Results', fontsize=13)
    plt.tight_layout()
    plt.savefig(run_dir / 'results.png', dpi=150, bbox_inches='tight')
    plt.close()


def _save_curve(history, key, name, run_dir):
    ep = [r['epoch'] for r in history]
    plt.figure(figsize=(7, 4))
    plt.plot(ep, [r[key] for r in history], linewidth=2)
    plt.title(f'{name} Curve')
    plt.xlabel('Época')
    plt.ylabel(f'{name} (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / f'{name}_curve.png', dpi=120)
    plt.close()


def _save_pr_curve(history, run_dir):
    ep = [r['epoch'] for r in history]
    plt.figure(figsize=(7, 4))
    plt.plot([r['metrics/recall'] for r in history],
             [r['metrics/precision'] for r in history], linewidth=2)
    plt.xlabel('Recall (%)')
    plt.ylabel('Precision (%)')
    plt.title('PR Curve (por época)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / 'PR_curve.png', dpi=120)
    plt.close()


def save_confusion_matrices(y_true, y_pred, run_dir):
    """Salva as matrizes de confusão (absoluta e normalizada) do run."""
    cm     = confusion_matrix(y_true, y_pred)
    cm_n   = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    for mat, fname, fmt in [(cm, 'confusion_matrix.png', 'd'),
                             (cm_n, 'confusion_matrix_n.png', '.2f')]:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(mat, annot=True, fmt=fmt, cmap='Blues', cbar=False,
                    annot_kws={'size': 20},
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_xlabel('Predito', fontsize=13)
        ax.set_ylabel('Real',    fontsize=13)
        ax.tick_params(labelsize=12)
        plt.tight_layout()
        plt.savefig(run_dir / fname, dpi=120)
        plt.close()


# ── Utilitário de run dir ─────────────────────────────────────────────────────

def _next_run_dir(save_dir: str, name: str) -> Path:
    base = Path(save_dir) / name
    if not base.exists():
        return base
    i = 2
    while (Path(save_dir) / f'{name}{i}').exists():
        i += 1
    return Path(save_dir) / f'{name}{i}'


# ── Orquestrador principal ────────────────────────────────────────────────────

def train_model(model, loaders, model_name: str, condition: str,
                device, logger, save_dir='runs'):
    train_loader, val_loader, test_loader = loaders
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    safe    = model_name.lower().replace('-', '').replace(' ', '')
    run_dir = _next_run_dir(save_dir, f'{safe}_{condition}')
    weights_dir = run_dir / 'weights'
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(exist_ok=True)

    best_path = weights_dir / 'best.pt'
    last_path = weights_dir / 'last.pt'

    # args.yaml
    with open(run_dir / 'args.yaml', 'w') as f:
        yaml.dump({
            'model': model_name, 'condition': condition,
            'epochs': EPOCHS, 'lr': LR, 'optimizer': 'Adam',
            'loss': 'CrossEntropyLoss', 'batch': train_loader.batch_size,
            'seed': 42, 'classes': CLASSES,
        }, f, default_flow_style=False)

    # results.csv
    CSV_FIELDS = ['epoch', 'train/loss', 'train/acc', 'val/loss', 'val/acc',
                  'metrics/precision', 'metrics/recall', 'metrics/f1', 'lr']
    csv_f = open(run_dir / 'results.csv', 'w', newline='', encoding='utf-8')
    csv_w = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
    csv_w.writeheader()

    best_val_acc = 0.0
    history      = []

    # Cabeçalho
    HDR = (f'{"Época":>10}  {"GPU":>5}  {"Tr_Loss":>8}  {"Tr_Acc":>8}  '
           f'{"Vl_Loss":>8}  {"Vl_Acc":>8}  {"Prec":>7}  {"Rec":>7}  {"F1":>7}')
    print(f'\n  {model_name} | {condition.upper()}  →  {run_dir}')
    print(f'  {HDR}')
    print(f'  {"─" * len(HDR)}')

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = _train_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, EPOCHS, run_dir)

        vl_loss, vl_acc, vl_true, vl_pred = _evaluate(
            model, val_loader, criterion, device,
            desc=f'  {epoch:3d}/{EPOCHS} [val]   ')

        tr_acc_pct  = tr_acc  * 100
        tr_loss_pct = tr_loss * 100
        vl_acc_pct  = vl_acc  * 100
        vl_loss_pct = vl_loss * 100
        prec = precision_score(vl_true, vl_pred, zero_division=0) * 100
        rec  = recall_score(vl_true,   vl_pred, zero_division=0) * 100
        f1   = f1_score(vl_true,       vl_pred, zero_division=0) * 100

        torch.save(model.state_dict(), last_path)
        is_best = vl_acc_pct > best_val_acc
        if is_best:
            best_val_acc = vl_acc_pct
            torch.save(model.state_dict(), best_path)

        print(f'  {epoch:3d}/{EPOCHS}  {_gpu_mem():>5}  '
              f'{tr_loss_pct:7.2f}%  {tr_acc_pct:7.2f}%  '
              f'{vl_loss_pct:7.2f}%  {vl_acc_pct:7.2f}%  '
              f'{prec:6.2f}%  {rec:6.2f}%  {f1:6.2f}%'
              + ('  ★' if is_best else ''))

        row = {
            'epoch': epoch,
            'train/loss': round(tr_loss_pct, 4), 'train/acc': round(tr_acc_pct, 4),
            'val/loss':   round(vl_loss_pct, 4), 'val/acc':   round(vl_acc_pct, 4),
            'metrics/precision': round(prec, 4),
            'metrics/recall':    round(rec,  4),
            'metrics/f1':        round(f1,   4),
            'lr': LR,
        }
        csv_w.writerow(row)
        csv_f.flush()
        history.append(row)

        logger.log_epoch(epoch, EPOCHS, model_name, condition,
                         tr_acc_pct, tr_loss_pct, vl_acc_pct, vl_loss_pct)

    csv_f.close()

    # Teste com best weights
    print(f'\n  Carregando best weights...')
    model.load_state_dict(torch.load(best_path, map_location=device))
    te_loss, te_acc, y_true, y_pred = _evaluate(
        model, test_loader, criterion, device, desc='  [teste final]',
        run_dir=run_dir, batch_prefix='val_batch')

    te_acc_pct  = te_acc  * 100
    te_loss_pct = te_loss * 100
    logger.log_test(model_name, condition, te_acc_pct, te_loss_pct)

    te_prec = precision_score(y_true, y_pred, zero_division=0) * 100
    te_rec  = recall_score(y_true,   y_pred, zero_division=0) * 100
    te_f1   = f1_score(y_true,       y_pred, zero_division=0) * 100

    with open(run_dir / 'test_results.yaml', 'w') as f:
        yaml.dump({
            'test_acc':       round(te_acc_pct,  4),
            'test_loss':      round(te_loss_pct, 4),
            'test_precision': round(te_prec, 4),
            'test_recall':    round(te_rec,  4),
            'test_f1':        round(te_f1,   4),
        }, f, default_flow_style=False)

    # Plots
    print('  Gerando plots...')
    _save_results_png(history, run_dir)
    _save_curve(history, 'metrics/f1',        'F1',        run_dir)
    _save_curve(history, 'metrics/precision',  'Precision', run_dir)
    _save_curve(history, 'metrics/recall',     'Recall',    run_dir)
    _save_pr_curve(history, run_dir)
    save_confusion_matrices(y_true, y_pred, run_dir)

    print(f'\n  Run salvo em: {run_dir}')
    print(f'  weights/best.pt  (val_acc={best_val_acc:.2f}%)')
    print(f'  weights/last.pt\n')

    return history, {'test_acc': te_acc_pct, 'test_loss': te_loss_pct,
                     'y_true': y_true, 'y_pred': y_pred}
