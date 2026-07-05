"""
Grad-CAM — imagens retiradas automaticamente do split de teste.
Uso: python gradcam.py
     python gradcam.py --runs runs --out gradcam_output
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import importlib
import numpy as np
import pandas as pd
import torch
import cv2
import yaml
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ALL_MODELS     = ['ResNet-18', 'ResNet-34', 'ResNet-50', 'MobileNetV2', 'EfficientNet-B0', 'AlexNet', 'VGG-16']
_ALL_CONDITIONS = ['base', 'clahe', 'he', 'gamma', 'stretch']
CLASSES = ['Normal', 'Tumor']
SEED    = 42

_MODEL_MODULE = {
    'ResNet-18':       'model_resnet18',
    'ResNet-34':       'model_resnet34',
    'ResNet-50':       'model_resnet50',
    'MobileNetV2':     'model_mobilenetv2',
    'EfficientNet-B0': 'model_efficientnetb0',
    'AlexNet':         'model_alexnet',
    'VGG-16':          'model_vgg16',
}

_TARGET_LAYER = {
    'ResNet-18':       lambda m: m.layer4[-1],
    'ResNet-34':       lambda m: m.layer4[-1],
    'ResNet-50':       lambda m: m.layer4[-1],
    'MobileNetV2':     lambda m: m.features[-1],
    'EfficientNet-B0': lambda m: m.features[-1],
    'AlexNet':         lambda m: m.features[10],   # último Conv2d (256 canais)
    'VGG-16':          lambda m: m.features[28],   # último Conv2d (512 canais)
}

_IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ── Menu interativo ────────────────────────────────────────────────────────────

def _select(label: str, options: list) -> list:
    print(f'\n  {label}:')
    print('    [0] Todos')
    for i, opt in enumerate(options, 1):
        print(f'    [{i}] {opt}')
    while True:
        raw = input('  Selecione (ex: 0  ou  1,2): ').strip()
        if raw == '0':
            return list(options)
        try:
            idxs = [int(x.strip()) for x in raw.split(',')]
            if idxs and all(1 <= idx <= len(options) for idx in idxs):
                seen = set()
                return [options[idx - 1] for idx in idxs
                        if not (options[idx - 1] in seen or seen.add(options[idx - 1]))]
        except ValueError:
            pass
        print('  Entrada inválida. Tente novamente.')


def _ask_n_images() -> int:
    while True:
        raw = input('\n  Quantas imagens por modelo/condição? [ex: 4]: ').strip()
        try:
            n = int(raw)
            if n > 0:
                return n
        except ValueError:
            pass
        print('  Digite um número inteiro positivo.')


def _prompt() -> tuple:
    print('\n' + '═' * 52)
    print('  Grad-CAM — seleção de experimentos')
    print('═' * 52)
    models     = _select('Modelos',   _ALL_MODELS)
    conditions = _select('Condições', _ALL_CONDITIONS)
    n_images   = _ask_n_images()

    print(f'\n  Combinações ({len(models) * len(conditions)}):')
    for c in conditions:
        for m in models:
            print(f'    • {m} — {c}')
    print(f'  Imagens por combinação: {n_images} (balanceadas: Normal + Tumor)\n')

    confirm = input('  Confirmar? [s/N]: ').strip().lower()
    if confirm != 's':
        print('  Cancelado.')
        raise SystemExit(0)
    return models, conditions, n_images


# ── Run finder ─────────────────────────────────────────────────────────────────

def _find_run(model_name: str, condition: str, runs_dir: Path) -> Path | None:
    safe      = model_name.lower().replace('-', '').replace(' ', '')
    base_name = f'{safe}_{condition}'
    candidates = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        suffix = d.name[len(base_name):]
        if d.name == base_name or (d.name.startswith(base_name) and suffix.isdigit()):
            candidates.append(d)
    if not candidates:
        return None
    candidates.sort(key=lambda d: int(d.name[len(base_name):]) if d.name != base_name else 0,
                    reverse=True)
    run = candidates[0]
    return run if (run / 'weights' / 'best.pt').exists() else None


# ── Split de teste (mesma lógica do dataset.py) ────────────────────────────────

def _get_test_samples(dataset_path: str, csv_path: str, n: int) -> pd.DataFrame:
    from dataset import normalize_csv_path

    df = pd.read_csv(csv_path)
    df['path'] = df['path'].apply(lambda p: normalize_csv_path(p, dataset_path))
    df = df[df['Class'].isin(['Tumor', 'Normal'])].reset_index(drop=True)

    tumor_df  = df[df['Class'] == 'Tumor']
    normal_df = df[df['Class'] == 'Normal'].sample(n=len(tumor_df), random_state=SEED)
    balanced  = (pd.concat([tumor_df, normal_df])
                 .sample(frac=1, random_state=SEED)
                 .reset_index(drop=True))

    def _split(sub):
        tr, tmp = train_test_split(sub, test_size=0.4, random_state=SEED)
        _, te   = train_test_split(tmp, test_size=0.5, random_state=SEED)
        return te

    n_te = _split(balanced[balanced['Class'] == 'Normal'])
    t_te = _split(balanced[balanced['Class'] == 'Tumor'])
    test_df = pd.concat([n_te, t_te]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Balanceia as amostras entre as classes
    half = n // 2
    rem  = n - half
    normals = test_df[test_df['Class'] == 'Normal'].sample(min(half, len(n_te)), random_state=SEED)
    tumors  = test_df[test_df['Class'] == 'Tumor' ].sample(min(rem,  len(t_te)), random_state=SEED)
    return pd.concat([normals, tumors]).sample(frac=1, random_state=SEED).reset_index(drop=True)


# ── Pré-processamento e tensor ─────────────────────────────────────────────────

def _to_input(img_path: str, condition: str) -> tuple[np.ndarray, torch.Tensor]:
    from dataset import _PREPROC
    img = Image.open(img_path).convert('L')
    fn  = _PREPROC.get(condition)
    if fn:
        img = fn(img)
    img_rgb = img.convert('RGB')
    img_vis = np.array(img_rgb.resize((224, 224)))
    tensor  = _IMG_TRANSFORM(img_rgb).unsqueeze(0)
    return img_vis, tensor


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self._activations = None

        def _fwd_hook(module, inp, out):
            out.retain_grad()       # preserva grad em tensor não-folha
            self._activations = out

        target_layer.register_forward_hook(_fwd_hook)

    def __call__(self, tensor, device):
        self.model.eval()
        # requires_grad no input cria o grafo de computação mesmo com backbone congelado
        tensor = tensor.to(device).requires_grad_(True)
        out    = self.model(tensor)
        probs  = torch.softmax(out, dim=1)[0]
        cls    = out.argmax(dim=1).item()
        conf   = probs[cls].item() * 100

        self.model.zero_grad()
        out[0, cls].backward()

        grads = self._activations.grad[0]          # (C, H, W)
        acts  = self._activations.detach()[0]      # (C, H, W)
        w     = grads.mean(dim=(1, 2))
        cam   = torch.relu((w[:, None, None] * acts).sum(0))
        cam   = cam.cpu().numpy()
        cam   = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, cls, conf


def _heatmap(cam: np.ndarray) -> np.ndarray:
    heat = cv2.applyColorMap(
        cv2.resize((cam * 255).astype(np.uint8), (224, 224)), cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def _overlay(img_vis: np.ndarray, heat: np.ndarray) -> np.ndarray:
    return (0.5 * img_vis + 0.5 * heat).clip(0, 255).astype(np.uint8)


# ── Figura ────────────────────────────────────────────────────────────────────

def _save_figure(samples_vis, samples_heat, samples_overlay, labels, preds, confs,
                 model_name, condition, out_dir: Path):
    n    = len(samples_vis)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.5 * n))
    if n == 1:
        axes = [axes]

    for i, (vis, heat, ov, true_cls, pred_cls, conf) in enumerate(
            zip(samples_vis, samples_heat, samples_overlay, labels, preds, confs)):
        color = 'green' if true_cls == pred_cls else 'red'

        axes[i][0].imshow(vis)
        axes[i][0].set_title(f'Original\nReal: {CLASSES[true_cls]}', fontsize=9)
        axes[i][0].axis('off')

        axes[i][1].imshow(heat)
        axes[i][1].set_title('Heatmap', fontsize=9)
        axes[i][1].axis('off')

        axes[i][2].imshow(ov)
        axes[i][2].set_title(f'Sobreposição\nPred: {CLASSES[pred_cls]} ({conf:.1f}%)',
                              fontsize=9, color=color)
        axes[i][2].axis('off')

    fig.suptitle(f'{model_name}  |  {condition}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    safe = model_name.lower().replace('-', '').replace(' ', '')
    path = out_dir / f'gradcam_{safe}_{condition}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ── Auto-detect head architecture from saved weights ───────────────────────────

def _build_matching(model_name: str, state_dict: dict):
    """
    Constrói o modelo com a cabeça correta para o state_dict salvo.
    Trata o caso em que runs antigos foram salvos com MLP em ResNets
    (período antes de reverter para linear simples).
    """
    import importlib
    import torch.nn as nn
    from torchvision import models as tvm

    # Detecta se a cabeça salva é MLP (fc.0) ou linear simples (fc.weight)
    has_mlp_fc = any(k.startswith('fc.0') for k in state_dict)

    if has_mlp_fc and model_name in ('ResNet-18', 'ResNet-34', 'ResNet-50'):
        _weights_map = {
            'ResNet-18': (tvm.resnet18,  tvm.ResNet18_Weights.IMAGENET1K_V1),
            'ResNet-34': (tvm.resnet34,  tvm.ResNet34_Weights.IMAGENET1K_V1),
            'ResNet-50': (tvm.resnet50,  tvm.ResNet50_Weights.IMAGENET1K_V1),
        }
        fn, w = _weights_map[model_name]
        m = fn(weights=w)
        for p in m.parameters():
            p.requires_grad = False
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Linear(in_f, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
        return m

    mod = importlib.import_module(_MODEL_MODULE[model_name])
    return mod.build()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='runs')
    parser.add_argument('--out',  default='gradcam_output')
    cli = parser.parse_args()

    selected_models, selected_conditions, n_images = _prompt()

    runs_dir = Path(cli.runs)
    out_dir  = Path(cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    import kagglehub
    try:
        dl_path = kagglehub.dataset_download(
            'nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone')
    except Exception:
        dl_path = os.path.expanduser(
            '~/.cache/kagglehub/datasets/nazmul0087/'
            'ct-kidney-dataset-normal-cyst-tumor-and-stone/versions/1')
    csv_path     = os.path.join(dl_path, 'KidneyData.csv')
    dataset_path = os.path.join(dl_path,
                                'CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone',
                                'CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')

    samples_df = _get_test_samples(dataset_path, csv_path, n_images)
    print(f'Amostras do teste: {len(samples_df)} '
          f'({(samples_df["Class"]=="Normal").sum()} Normal, '
          f'{(samples_df["Class"]=="Tumor").sum()} Tumor)\n')

    _LABEL_MAP = {'Normal': 0, 'Tumor': 1}

    for condition in selected_conditions:
        for model_name in selected_models:
            run_dir = _find_run(model_name, condition, runs_dir)
            if run_dir is None:
                print(f'  [SKIP] {model_name} / {condition} — run não encontrado')
                continue

            print(f'  {model_name} — {condition}  ({run_dir.name})')
            best_path  = run_dir / 'weights' / 'best.pt'
            state_dict = torch.load(best_path, map_location=device)
            model      = _build_matching(model_name, state_dict).to(device)
            model.load_state_dict(state_dict)

            gcam = GradCAM(model, _TARGET_LAYER[model_name](model))

            vis_list, heat_list, ov_list, label_list, pred_list, conf_list = [], [], [], [], [], []
            for _, row in samples_df.iterrows():
                img_vis, tensor = _to_input(row['path'], condition)
                cam, pred, conf = gcam(tensor, device)
                heat = _heatmap(cam)
                vis_list.append(img_vis)
                heat_list.append(heat)
                ov_list.append(_overlay(img_vis, heat))
                label_list.append(_LABEL_MAP[row['Class']])
                pred_list.append(pred)
                conf_list.append(conf)

            out_path = _save_figure(vis_list, heat_list, ov_list,
                                    label_list, pred_list, conf_list,
                                    model_name, condition, out_dir)
            print(f'    → salvo em {out_path}')

            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    print(f'\nProntos em: {out_dir}/')


if __name__ == '__main__':
    main()
