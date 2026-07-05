# Targets do artigo RITA-VOL32-NR1-61 — Tabela 1
# Todos os valores em percentual (%)
ARTICLE_TARGETS = {
    'ResNet-50': {
        'train_acc':  91.10,
        'val_acc':    92.90,
        'test_acc':   95.45,
        'train_loss': 26.57,
        'val_loss':   24.49,
        'test_loss':  20.50,
    },
    'MobileNetV2': {
        'train_acc':  96.75,
        'val_acc':    98.78,
        'test_acc':   99.89,
        'train_loss': 10.20,
        'val_loss':    6.10,
        'test_loss':   1.94,
    },
}

TOLERANCE = 2.0  # ±2% considerado "dentro do esperado"
