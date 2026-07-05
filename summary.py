"""
Lê todos os runs em runs/ e exibe tabela de métricas de treino e teste.
Uso: python summary.py
     python summary.py --runs /caminho/para/runs
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import argparse
import yaml
import pandas as pd
from pathlib import Path


def _read_run(run_dir: Path) -> dict | None:
    args_path = run_dir / 'args.yaml'
    csv_path  = run_dir / 'results.csv'
    if not args_path.exists() or not csv_path.exists():
        return None

    with open(args_path) as f:
        args = yaml.safe_load(f)

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    best_idx  = df['val/acc'].idxmax()
    last      = df.iloc[-1]
    best_val  = df.loc[best_idx]

    row = {
        'Run':              run_dir.name,
        'Modelo':           args.get('model',     '?'),
        'Condição':         args.get('condition', '?'),
        'Épocas':           len(df),
        'Train Acc':        round(last['train/acc'],      2),
        'Train Loss':       round(last['train/loss'],     2),
        'Val Acc (melhor)': round(best_val['val/acc'],    2),
        'Val Loss (melhor)':round(best_val['val/loss'],   2),
        'Val Época melhor': int(best_val['epoch']),
        'Test Acc':         None,
        'Test Loss':        None,
        'Test Prec':        None,
        'Test Recall':      None,
        'Test F1':          None,
    }

    test_path = run_dir / 'test_results.yaml'
    if test_path.exists():
        with open(test_path) as f:
            t = yaml.safe_load(f)
        row['Test Acc']    = t.get('test_acc')
        row['Test Loss']   = t.get('test_loss')
        row['Test Prec']   = t.get('test_precision')
        row['Test Recall'] = t.get('test_recall')
        row['Test F1']     = t.get('test_f1')

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='runs', help='Pasta com os runs')
    args = parser.parse_args()

    runs_dir = Path(args.runs)
    if not runs_dir.exists():
        print(f'Pasta "{runs_dir}" não encontrada.')
        return

    rows = []
    for d in sorted(runs_dir.iterdir()):
        if d.is_dir():
            r = _read_run(d)
            if r:
                rows.append(r)

    if not rows:
        print('Nenhum run com resultados encontrado.')
        return

    df = pd.DataFrame(rows)

    # ── Tabela de treino/val ──────────────────────────────────────────────────
    print('\n' + '═' * 90)
    print('  TREINO / VALIDAÇÃO')
    print('═' * 90)
    tv = df[['Run', 'Modelo', 'Condição', 'Épocas',
             'Train Acc', 'Train Loss',
             'Val Acc (melhor)', 'Val Loss (melhor)', 'Val Época melhor']].copy()
    for col in ['Train Acc', 'Train Loss', 'Val Acc (melhor)', 'Val Loss (melhor)']:
        tv[col] = tv[col].apply(lambda x: f'{x:.2f}%' if x is not None else 'N/A')
    print(tv.to_string(index=False))

    # ── Tabela de teste ───────────────────────────────────────────────────────
    print('\n' + '═' * 90)
    print('  TESTE (best weights)')
    print('═' * 90)
    te = df[['Run', 'Modelo', 'Condição',
             'Test Acc', 'Test Loss',
             'Test Prec', 'Test Recall', 'Test F1']].copy()
    for col in ['Test Acc', 'Test Loss', 'Test Prec', 'Test Recall', 'Test F1']:
        te[col] = te[col].apply(lambda x: f'{x:.2f}%' if x is not None else 'N/A')
    print(te.to_string(index=False))
    print('═' * 90 + '\n')


if __name__ == '__main__':
    main()
