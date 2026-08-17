"""
Reavalia todos os runs existentes no conjunto de teste usando best.pt.
Salva test_results.yaml em cada run e exibe tabela final.
Uso: python retest.py
     python retest.py --runs /caminho/runs
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Inferência determinística: sem isso a seleção de kernels do cuDNN pode fazer
# amostras na fronteira de decisão trocarem de classe entre execuções.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

_MODEL_BUILDERS = {
    'ResNet-18': 'model_resnet18',
    'ResNet-34': 'model_resnet34',
    'ResNet-50': 'model_resnet50',
    'AlexNet':   'model_alexnet',
    'VGG-16':    'model_vgg16',
}


def _build_model(model_name: str):
    import importlib
    module_name = _MODEL_BUILDERS.get(model_name)
    if module_name is None:
        raise ValueError(f'Modelo desconhecido: {model_name}')
    mod = importlib.import_module(module_name)
    return mod.build()


def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for n, (imgs, labels) in enumerate(loader, 1):
            out   = model(imgs.to(device))
            loss  = criterion(out, labels.to(device, dtype=torch.long))
            preds = out.argmax(dim=1).cpu()
            total_loss += loss.item()
            correct    += (preds == labels.long()).sum().item()
            total      += labels.size(0)
            all_preds.extend(preds.numpy().astype(int))
            all_labels.extend(labels.numpy().astype(int))
    loss_pct = total_loss / len(loader) * 100
    acc_pct  = correct / total * 100
    return acc_pct, loss_pct, np.array(all_labels), np.array(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs',    default='runs')
    parser.add_argument('--csv',     default=None, help='Caminho para KidneyData.csv')
    parser.add_argument('--dataset', default=None, help='Caminho para pasta do dataset')
    args = parser.parse_args()

    import kagglehub
    import dataset as ds
    import trainer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    # ── Dataset path ──────────────────────────────────────────────────────────
    if args.csv and args.dataset:
        csv_path     = args.csv
        dataset_path = args.dataset
    else:
        try:
            dl_path = kagglehub.dataset_download(
                'nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone')
        except Exception:
            dl_path = os.path.expanduser(
                '~/.cache/kagglehub/datasets/nazmul0087/'
                'ct-kidney-dataset-normal-cyst-tumor-and-stone/versions/1')
        csv_path     = os.path.join(dl_path, 'kidneyData.csv')
        dataset_path = os.path.join(dl_path,
                                    'CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone',
                                    'CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone')

    criterion = nn.CrossEntropyLoss()
    runs_dir  = Path(args.runs)
    rows      = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        args_path = run_dir / 'args.yaml'
        best_path = run_dir / 'weights' / 'best.pt'
        if not args_path.exists() or not best_path.exists():
            continue

        with open(args_path) as f:
            cfg = yaml.safe_load(f)

        model_name = cfg.get('model')
        condition  = cfg.get('condition')

        # Runs fora do escopo do artigo (outras arquiteturas ou condições)
        if model_name not in _MODEL_BUILDERS or condition not in ds._PREPROC:
            print(f'Pulando    {run_dir.name}  ({model_name} / {condition}) '
                  f'— fora do escopo do artigo')
            continue

        print(f'Avaliando  {run_dir.name}  ({model_name} / {condition})...')

        try:
            _, _, test_loader = ds.make_loaders(
                dataset_path, csv_path, condition=condition, device=device)

            model = _build_model(model_name).to(device)
            model.load_state_dict(torch.load(best_path, map_location=device))

            acc, loss, y_true, y_pred = _evaluate(model, test_loader, criterion, device)
            prec = precision_score(y_true, y_pred, zero_division=0) * 100
            rec  = recall_score(y_true,   y_pred, zero_division=0) * 100
            f1   = f1_score(y_true,       y_pred, zero_division=0) * 100
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            result = {
                'test_acc':       round(acc,  4),
                'test_loss':      round(loss, 4),
                'test_precision': round(prec, 4),
                'test_recall':    round(rec,  4),
                'test_f1':        round(f1,   4),
                # Contagens da matriz de confusão (classe positiva: Tumor).
                # Persistidas para que as figuras sejam geradas a partir dos
                # mesmos números da tabela, sem depender de PNGs rasterizados.
                'test_tn': int(tn), 'test_fp': int(fp),
                'test_fn': int(fn), 'test_tp': int(tp),
            }
            with open(run_dir / 'test_results.yaml', 'w') as f:
                yaml.dump(result, f, default_flow_style=False)

            # Regera as matrizes do run para manterem-se coerentes com o YAML
            trainer.save_confusion_matrices(y_true, y_pred, run_dir)

            rows.append({'Run': run_dir.name, 'Modelo': model_name,
                         'Condição': condition, **result})
            print(f'  Test Acc={acc:.2f}%  Prec={prec:.2f}%  Rec={rec:.2f}%  F1={f1:.2f}%')

            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            print(f'  ERRO: {e}')

    if not rows:
        print('Nenhum run avaliado.')
        return

    df = pd.DataFrame(rows)
    for col in ['test_acc', 'test_loss', 'test_precision', 'test_recall', 'test_f1']:
        df[col] = df[col].apply(lambda x: f'{x:.2f}%')

    df = df[['Run', 'Modelo', 'Condição',
             'test_acc', 'test_loss', 'test_precision', 'test_recall', 'test_f1',
             'test_tn', 'test_fp', 'test_fn', 'test_tp']]
    df.columns = ['Run', 'Modelo', 'Condição',
                  'Test Acc', 'Test Loss', 'Precision', 'Recall', 'F1',
                  'TN', 'FP', 'FN', 'TP']

    print('\n' + '═' * 95)
    print('  RESULTADO DE TESTE — todos os runs')
    print('═' * 95)
    print(df.to_string(index=False))
    print('═' * 95)


if __name__ == '__main__':
    main()
