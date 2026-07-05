import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SEED     = 42
IMG_SIZE = 224
BATCH    = 32

_LABEL_MAP = {'Normal': 0, 'Tumor': 1}
_COLAB_PREFIX = '/content/data/CT KIDNEY DATASET Normal, CYST, TUMOR and STONE'

_clahe_proc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def normalize_csv_path(p: str, dataset_path: str) -> str:
    p = p.replace(_COLAB_PREFIX, dataset_path)
    p = (p.replace('/TUMOR/', '/Tumor/')
           .replace('/NORMAL/', '/Normal/')
           .replace('/CYST/', '/Cyst/')
           .replace('/STONE/', '/Stone/'))
    return p


def _apply_clahe(pil_gray: Image.Image) -> Image.Image:
    arr = np.array(pil_gray)
    return Image.fromarray(_clahe_proc.apply(arr))


def _apply_he(pil_gray: Image.Image) -> Image.Image:
    arr = np.array(pil_gray)
    return Image.fromarray(cv2.equalizeHist(arr))


def _apply_gamma(pil_gray: Image.Image, gamma: float = 0.7) -> Image.Image:
    arr = np.array(pil_gray).astype(np.float32) / 255.0
    arr = np.power(arr, gamma) * 255.0
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def _apply_stretch(pil_gray: Image.Image) -> Image.Image:
    arr = np.array(pil_gray).astype(np.float32)
    p2, p98 = np.percentile(arr, (2, 98))
    arr = np.clip((arr - p2) / (p98 - p2 + 1e-8) * 255.0, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

_PREPROC = {
    'base':    None,
    'clahe':   _apply_clahe,
    'he':      _apply_he,
    'gamma':   _apply_gamma,
    'stretch': _apply_stretch,
}


class KidneyDataset(Dataset):
    def __init__(self, df, condition: str = 'base'):
        self.df      = df.reset_index(drop=True)
        self.preproc = _PREPROC[condition]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = _LABEL_MAP[row['Class']]
        img   = Image.open(row['path']).convert('L')
        if self.preproc is not None:
            img = self.preproc(img)
        img = img.convert('RGB')
        return _transform(img), torch.tensor(label, dtype=torch.float32)


def _split_by_class(df_class):
    train, temp = train_test_split(df_class, test_size=0.4, random_state=SEED)
    val, test   = train_test_split(temp,     test_size=0.5, random_state=SEED)
    return train, val, test


def make_loaders(dataset_path: str, csv_path: str,
                 condition: str = 'base', device=None):
    """
    Lê o CSV, corrige paths, filtra Normal+Tumor, balanceia e faz split 60/20/20.
    condition: 'base' | 'clahe' | 'he'
    """
    if condition not in _PREPROC:
        raise ValueError(f"condition deve ser um de {list(_PREPROC)}, recebeu '{condition}'")

    df = pd.read_csv(csv_path)
    df['path'] = df['path'].apply(lambda p: normalize_csv_path(p, dataset_path))
    df = df[df['Class'].isin(['Tumor', 'Normal'])].reset_index(drop=True)

    tumor_df  = df[df['Class'] == 'Tumor']
    normal_df = df[df['Class'] == 'Normal'].sample(n=len(tumor_df), random_state=SEED)
    balanced  = (pd.concat([tumor_df, normal_df])
                 .sample(frac=1, random_state=SEED)
                 .reset_index(drop=True))

    n_tr, n_val, n_te = _split_by_class(balanced[balanced['Class'] == 'Normal'])
    t_tr, t_val, t_te = _split_by_class(balanced[balanced['Class'] == 'Tumor'])

    def _merge(a, b):
        return pd.concat([a, b]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    train_df = _merge(n_tr,  t_tr)
    val_df   = _merge(n_val, t_val)
    test_df  = _merge(n_te,  t_te)

    print(f'  Split — Treino: {len(train_df)} | Val: {len(val_df)} | Teste: {len(test_df)}')

    pin = device is not None and device.type == 'cuda'

    def _loader(split_df, shuffle):
        return DataLoader(
            KidneyDataset(split_df, condition),
            batch_size=BATCH, shuffle=shuffle,
            pin_memory=pin, num_workers=0,
        )

    return _loader(train_df, True), _loader(val_df, False), _loader(test_df, False)
