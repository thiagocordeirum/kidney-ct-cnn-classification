"""
Reprodução do artigo RITA-VOL32-NR1-61 + estudo de pré-processamento
Modelos   : ResNet-50 e MobileNetV2
Condições : base (sem pré-proc) | clahe | he (equalização global)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import numpy as np

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

_ALL_MODELS = ['ResNet-18', 'ResNet-34', 'ResNet-50', 'MobileNetV2', 'EfficientNet-B0', 'AlexNet', 'VGG-16']
_ALL_CONDITIONS = ['base', 'clahe', 'he', 'gamma', 'stretch']


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


def _prompt_experiments() -> tuple[list, list]:
    print('\n' + '═' * 52)
    print('  PIBIC — Seleção de experimentos')
    print('═' * 52)
    models     = _select('Modelos',    _ALL_MODELS)
    conditions = _select('Condições',  _ALL_CONDITIONS)
    print(f'\n  Experimentos selecionados ({len(models) * len(conditions)} treinos):')
    for c in conditions:
        for m in models:
            print(f'    • {m} — {c}')
    print()
    confirm = input('  Confirmar e iniciar? [s/N]: ').strip().lower()
    if confirm != 's':
        print('  Cancelado.')
        raise SystemExit(0)
    return models, conditions


def main():
    import kagglehub
    import dataset
    import trainer
    from logger import EpochLogger
    import model_resnet18
    import model_resnet34
    import model_resnet50
    import model_mobilenetv2
    import model_efficientnetb0
    import model_alexnet
    import model_vgg16

    _MODEL_FACTORIES = {
        'ResNet-18':       model_resnet18.build,
        'ResNet-34':       model_resnet34.build,
        'ResNet-50':       model_resnet50.build,
        'MobileNetV2':     model_mobilenetv2.build,
        'EfficientNet-B0': model_efficientnetb0.build,
        'AlexNet':         model_alexnet.build,
        'VGG-16':          model_vgg16.build,
    }


    selected_models, selected_conditions = _prompt_experiments()

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')
    if device.type == 'cuda':
        print(f'GPU   : {torch.cuda.get_device_name(0)}')
        torch.cuda.manual_seed(SEED)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print('\nVerificando dataset (kagglehub cache)...')
    try:
        dl_path = kagglehub.dataset_download('nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone')
    except Exception as e:
        print(f'  API Kaggle indisponível ({e.__class__.__name__}) — usando cache local.')
        dl_path = os.path.expanduser(
            '~/.cache/kagglehub/datasets/nazmul0087/'
            'ct-kidney-dataset-normal-cyst-tumor-and-stone/versions/1'
        )
    csv_path     = os.path.join(dl_path, 'kidneyData.csv')
    dataset_path = os.path.join(dl_path,
                                'CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone',
                                'CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone')
    print(f'Path  : {dl_path}')

    # ── Loop de experimentos ──────────────────────────────────────────────────
    logger      = EpochLogger(results_dir='results')
    all_results = {}

    for condition in selected_conditions:
        print(f'\n{"="*68}')
        print(f'  Condição: {condition.upper()}')
        print(f'{"="*68}')
        print(f'\n  Criando DataLoaders...')
        loaders = dataset.make_loaders(dataset_path, csv_path,
                                       condition=condition, device=device)

        for model_name in selected_models:
            print(f'\n  Treinando: {model_name} — {condition.upper()}')
            model = _MODEL_FACTORIES[model_name]().to(device)
            history, result = trainer.train_model(
                model, loaders, model_name, condition, device, logger,
                save_dir='runs',
            )
            all_results[(model_name, condition)] = result

            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    logger.close()

    # ── Resumo final ──────────────────────────────────────────────────────────
    from targets import ARTICLE_TARGETS, TOLERANCE
    print('\n' + '=' * 68)
    print('  RESUMO FINAL')
    print('=' * 68)
    print(f'  {"Modelo":<14} {"Cond":>6} {"Test Acc":>9} {"Artigo":>9} {"Delta":>8}  Status')
    print(f'  {"-"*60}')
    for (name, cond), res in all_results.items():
        tgt    = ARTICLE_TARGETS.get(name, {}) if cond == 'base' else {}
        artigo = tgt.get('test_acc', None)
        if artigo is not None:
            delta = res['test_acc'] - artigo
            ok = '✅' if abs(delta) <= TOLERANCE else ('⚠️ ' if abs(delta) <= TOLERANCE * 2 else '❌')
            print(f'  {name:<14} {cond:>6} {res["test_acc"]:>8.2f}%  '
                  f'{artigo:>7}%  {delta:>+7.2f}%  {ok}')
        else:
            print(f'  {name:<14} {cond:>6} {res["test_acc"]:>8.2f}%  {"--":>9}   {"--":>8}')
    print('=' * 68)
    print('\nRuns salvas em: runs/')


if __name__ == '__main__':
    main()
