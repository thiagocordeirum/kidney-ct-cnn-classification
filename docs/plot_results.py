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

import yaml
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CLASSES        = ['Normal', 'Tumor']
N_POR_CLASSE   = 457          # conjunto de teste balanceado: 914 amostras

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
    ('Gama',  'gamma'),
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

def load_counts(slug: str, cond: str):
    """
    Lê (TN, FP, FN, TP) do test_results.yaml do run.

    As contagens são gravadas pelo retest.py. Para runs avaliados antes dessa
    mudança, deriva-as de precisão e recall — exato, pois o conjunto de teste
    é balanceado (457 amostras por classe).
    """
    path = RUNS_DIR / f'{slug}_{cond}' / 'test_results.yaml'
    if not path.exists():
        print(f'  [AVISO] não encontrado: {path}')
        return None

    d = yaml.safe_load(path.read_text(encoding='utf-8'))
    if all(k in d for k in ('test_tn', 'test_fp', 'test_fn', 'test_tp')):
        return d['test_tn'], d['test_fp'], d['test_fn'], d['test_tp']

    prec, rec = d['test_precision'], d['test_recall']
    tp = round(rec / 100 * N_POR_CLASSE)
    fn = N_POR_CLASSE - tp
    fp = round(tp * (100 - prec) / prec) if prec > 0 else 0
    tn = N_POR_CLASSE - fp
    print(f'  [nota] {slug}_{cond}: contagens derivadas das métricas')
    return tn, fp, fn, tp


def compose_confusion():
    """
    Desenha as matrizes de confusão a partir das contagens.

    A versão anterior colava os PNGs de cada run, que ficavam ilegíveis após o
    reescalonamento para a largura da página.
    """
    # Proporção mantida próxima de 2,7:1 para que a figura ocupe a mesma
    # altura de coluna da versão anterior e a paginação do artigo não mude.
    n_cond, n_mod = len(CONDITIONS), len(MODELS)
    fig, axes = plt.subplots(
        n_cond, n_mod,
        figsize=(17.5, 5.4),
        gridspec_kw={'hspace': 0.16, 'wspace': 0.20},
    )

    for ci, (cond_label, cond_slug) in enumerate(CONDITIONS):
        for mi, (model_label, model_slug) in enumerate(MODELS):
            ax     = axes[ci, mi]
            counts = load_counts(model_slug, cond_slug)

            if counts is None:
                ax.set_facecolor('#eeeeee')
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=16, color='#888888')
                ax.set_xticks([]); ax.set_yticks([])
                continue

            tn, fp, fn, tp = counts
            cm = np.array([[tn, fp], [fn, tp]])

            ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())

            # Valores nas células, com contraste conforme o fundo
            limiar = cm.max() / 2
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f'{cm[i, j]:d}',
                            ha='center', va='center',
                            fontsize=23, fontweight='bold',
                            color='white' if cm[i, j] > limiar else '#1a1a1a')

            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(CLASSES, fontsize=15)
            ax.set_yticklabels(CLASSES, fontsize=15, rotation=90, va='center')
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)

            if ci == 0:
                ax.set_title(model_label, fontsize=18, fontweight='bold', pad=10)
            if ci == n_cond - 1:
                ax.set_xlabel('Predito', fontsize=15, labelpad=6)
            if mi == 0:
                # "Real" é rótulo de eixo (sem negrito); a condição identifica
                # a linha e vai à esquerda dele, em negrito.
                ax.set_ylabel('Real', fontsize=15, labelpad=8)
                ax.text(-0.42, 0.5, cond_label, transform=ax.transAxes,
                        fontsize=17, fontweight='bold', rotation=90,
                        ha='center', va='center')

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
