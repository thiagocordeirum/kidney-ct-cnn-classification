import csv
from pathlib import Path


class EpochLogger:
    """Registra métricas por época em CSV e imprime o progresso do treino."""

    def __init__(self, results_dir: str = 'results'):
        self.dir = Path(results_dir)
        self.dir.mkdir(exist_ok=True)
        self._files   = {}
        self._writers = {}

    def _writer(self, model_name: str, condition: str):
        key = f'{model_name}_{condition}'
        if key not in self._writers:
            fname = self.dir / f'{key.lower().replace("-","").replace(" ","")}_log.csv'
            f = open(fname, 'w', newline='', encoding='utf-8')
            w = csv.DictWriter(f, fieldnames=[
                'epoch', 'train_acc', 'train_loss', 'val_acc', 'val_loss'
            ])
            w.writeheader()
            self._files[key]   = f
            self._writers[key] = w
        return self._writers[key], self._files[key]

    def log_epoch(self, epoch: int, total: int, model_name: str,
                  condition: str, tr_acc: float, tr_loss: float,
                  vl_acc: float, vl_loss: float):
        """
        Imprime o progresso da época e registra a linha no CSV.
        Valores esperados em percentual (%).
        """
        print(
            f'[{model_name}] Ep {epoch:3d}/{total} | '
            f'Train Acc={tr_acc:6.2f}% Loss={tr_loss:6.2f}% | '
            f'Val Acc={vl_acc:6.2f}% Loss={vl_loss:6.2f}%'
        )

        w, f = self._writer(model_name, condition)
        w.writerow({
            'epoch':      epoch,
            'train_acc':  round(tr_acc,  4),
            'train_loss': round(tr_loss, 4),
            'val_acc':    round(vl_acc,  4),
            'val_loss':   round(vl_loss, 4),
        })
        f.flush()

    def log_test(self, model_name: str, condition: str,
                 test_acc: float, test_loss: float):
        sep = '=' * 68
        print(f'\n{sep}')
        print(f'  RESULTADO FINAL — {model_name} [{condition.upper()}]')
        print(sep)
        print(f'  Test Acc  = {test_acc:6.2f}%')
        print(f'  Test Loss = {test_loss:6.2f}%')
        print(f'{sep}\n')

    def close(self):
        for f in self._files.values():
            f.close()
