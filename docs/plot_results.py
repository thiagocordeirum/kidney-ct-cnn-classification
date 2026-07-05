"""
Gera as 3 figuras de resultados para o artigo:
  docs/fig_acc_curves.png  — curvas de acurácia (CLAHE e Gamma)
  docs/fig_loss_curves.png — curvas de perda    (CLAHE e Gamma)
  docs/fig_confusion.png   — matrizes de confusão compostas

Executar a partir de qualquer diretório:
  python docs/plot_results.py
ou de dentro de docs/:
  python plot_results.py
"""

import sys
from pathlib import Path

# garante que o script funcione de qualquer CWD
SCRIPT_DIR = Path(__file__).resolve().parent        # docs/
ROOT_DIR   = SCRIPT_DIR.parent                      # Etapa1/
RUNS_DIR   = ROOT_DIR / 'runs'
OUT_DIR    = SCRIPT_DIR

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Configuração ──────────────────────────────────────────────────────────────

MODELS = [
    ('ResNet-18', 'resnet18'),
    ('ResNet-34', 'resnet34'),
    ('ResNet-50', 'resnet50'),
    ('AlexNet',   'alexnet'),
    ('VGG-16',    'vgg16'),
]
CONDITIONS = [
    ('CLAHE', 'clahe'),
    ('Gamma', 'gamma'),
]

BLUE   = '#2E75B6'
ORANGE = '#E0702A'

plt.rcParams.update({
    'font.family':   'DejaVu Sans',
    'font.size':     8,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})


def load_csv(slug: str, cond: str) -> pd.DataFrame | None:
    """Carrega results.csv do run directory correspondente."""
    path = RUNS_DIR / f'{slug}_{cond}' / 'results.csv'
    if not path.exists():
        print(f'  [AVISO] não encontrado: {path}')
        return None
    return pd.read_csv(path)


def _label_condition_col(axes_row, cond_label: str):
    """Escreve o rótulo da condição à esquerda da linha."""
    axes_row[0].set_ylabel(cond_label, fontsize=9, fontweight='bold',
                           rotation=90, labelpad=6)


# ── Figura 1: Curvas de Acurácia ─────────────────────────────────────────────

def plot_acc_curves():
    n_cond = len(CONDITIONS)
    n_mod  = len(MODELS)
    fig, axes = plt.subplots(n_cond, n_mod,
                             figsize=(18, 5.5),
                             sharex=True,
                             gridspec_kw={'hspace': 0.35, 'wspace': 0.28})

    for ci, (cond_label, cond_slug) in enumerate(CONDITIONS):
        for mi, (model_label, model_slug) in enumerate(MODELS):
            ax  = axes[ci, mi]
            df  = load_csv(model_slug, cond_slug)

            if df is None:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='gray')
            else:
                ep = df['epoch']
                ax.plot(ep, df['train/acc'], color=BLUE,   linewidth=1.3,
                        label='Treino')
                ax.plot(ep, df['val/acc'],   color=ORANGE, linewidth=1.3,
                        label='Validação')
                ax.set_ylim(bottom=max(50, df['val/acc'].min() - 5))
                ax.set_ylim(top=101)

            # Título apenas na linha de cima
            if ci == 0:
                ax.set_title(model_label, fontweight='bold', pad=4)

            # Rótulo de condição na primeira coluna
            if mi == 0:
                ax.set_ylabel(f'{cond_label}\nAcurácia (%)',
                              fontsize=7.5, labelpad=4)

            # Eixo X apenas na última linha
            if ci == n_cond - 1:
                ax.set_xlabel('Época', fontsize=7)

            # Legenda apenas no primeiro painel de cada linha
            if mi == 0:
                ax.legend(loc='lower right')

    out = OUT_DIR / 'fig_acc_curves.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Salvo: {out}')


# ── Figura 2: Curvas de Perda ─────────────────────────────────────────────────

def plot_loss_curves():
    n_cond = len(CONDITIONS)
    n_mod  = len(MODELS)
    fig, axes = plt.subplots(n_cond, n_mod,
                             figsize=(18, 5.5),
                             sharex=True,
                             gridspec_kw={'hspace': 0.35, 'wspace': 0.28})

    for ci, (cond_label, cond_slug) in enumerate(CONDITIONS):
        for mi, (model_label, model_slug) in enumerate(MODELS):
            ax  = axes[ci, mi]
            df  = load_csv(model_slug, cond_slug)

            if df is None:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='gray')
            else:
                ep        = df['epoch']
                # loss está em ×100 no CSV; dividir por 100 → escala CE
                train_loss = df['train/loss'] / 100
                val_loss   = df['val/loss']   / 100
                ax.plot(ep, train_loss, color=BLUE,   linewidth=1.3,
                        label='Treino')
                ax.plot(ep, val_loss,   color=ORANGE, linewidth=1.3,
                        label='Validação')

            if ci == 0:
                ax.set_title(model_label, fontweight='bold', pad=4)

            if mi == 0:
                ax.set_ylabel(f'{cond_label}\nPerda (CE)',
                              fontsize=7.5, labelpad=4)

            if ci == n_cond - 1:
                ax.set_xlabel('Época', fontsize=7)

            if mi == 0:
                ax.legend(loc='upper right')

    out = OUT_DIR / 'fig_loss_curves.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Salvo: {out}')


# ── Figura 3: Matrizes de Confusão Compostas ──────────────────────────────────

def compose_confusion():
    n_cond = len(CONDITIONS)
    n_mod  = len(MODELS)

    fig, axes = plt.subplots(
        n_cond, n_mod,
        figsize=(18, 7),
        gridspec_kw={'hspace': 0.04, 'wspace': 0.04},
    )

    for ci, (cond_label, cond_slug) in enumerate(CONDITIONS):
        for mi, (model_label, model_slug) in enumerate(MODELS):
            ax = axes[ci, mi]
            img_path = RUNS_DIR / f'{model_slug}_{cond_slug}' / 'confusion_matrix.png'

            if img_path.exists():
                img = plt.imread(str(img_path))
                ax.imshow(img, aspect='auto')
            else:
                print(f'  [AVISO] matriz não encontrada: {img_path}')
                ax.set_facecolor('#eeeeee')
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=13, color='#888888')

            # Esconde ticks e bordas, mantém ylabel visível
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Nome do modelo — só na primeira linha
            if ci == 0:
                ax.set_title(model_label, fontsize=9, fontweight='bold', pad=5)

            # Rótulo da condição — só na primeira coluna, vertical
            if mi == 0:
                ax.set_ylabel(cond_label, fontsize=11, fontweight='bold',
                              rotation=90, labelpad=8)

    plt.subplots_adjust(left=0.06)
    out = OUT_DIR / 'fig_confusion.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Salvo: {out}')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Gerando figuras para o artigo...')
    print('\n[1/3] Curvas de acurácia...')
    plot_acc_curves()
    print('[2/3] Curvas de perda...')
    plot_loss_curves()
    print('[3/3] Matrizes de confusão...')
    compose_confusion()
    print('\nConcluído. Arquivos em docs/')
