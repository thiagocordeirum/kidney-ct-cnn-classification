import csv
from pathlib import Path
from targets import ARTICLE_TARGETS, TOLERANCE


def _status(value: float, target: float) -> str:
    delta = abs(value - target)
    if delta <= TOLERANCE:
        return '✅'
    elif delta <= TOLERANCE * 2:
        return '⚠️ '
    return '❌ '


class EpochLogger:
    """Registra métricas por época e compara com targets do artigo."""

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
        return self._writers[key], self._files[f'{model_name}_{condition}']

    def log_epoch(self, epoch: int, total: int, model_name: str,
                  condition: str, tr_acc: float, tr_loss: float,
                  vl_acc: float, vl_loss: float):
        """
        Imprime progresso e compara val_acc com target do artigo.
        Valores esperados em percentual (%).
        """
        tgt = ARTICLE_TARGETS.get(model_name, {})
        s_tr = _status(tr_acc, tgt.get('train_acc', tr_acc))
        s_vl = _status(vl_acc, tgt.get('val_acc',   vl_acc))

        print(
            f'[{model_name}] Ep {epoch:3d}/{total} | '
            f'Train Acc={tr_acc:6.2f}%{s_tr}(tgt {tgt.get("train_acc","--"):>5}) '
            f'Loss={tr_loss:6.2f}% | '
            f'Val Acc={vl_acc:6.2f}%{s_vl}(tgt {tgt.get("val_acc","--"):>5}) '
            f'Loss={vl_loss:6.2f}%'
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
        tgt  = ARTICLE_TARGETS.get(model_name, {})
        s_a  = _status(test_acc,  tgt.get('test_acc',  test_acc))
        s_l  = _status(test_loss, tgt.get('test_loss', test_loss))
        d_a  = test_acc  - tgt.get('test_acc',  test_acc)
        d_l  = test_loss - tgt.get('test_loss', test_loss)

        sep = '=' * 68
        print(f'\n{sep}')
        print(f'  RESULTADO FINAL — {model_name} [{condition.upper()}]')
        print(sep)
        print(f'  {"Métrica":<12} {"Obtido":>8}  {"Artigo":>8}  {"Delta":>8}  Status')
        print(f'  {"-"*52}')
        print(f'  {"Test Acc":<12} {test_acc:>7.2f}%  '
              f'{tgt.get("test_acc","--"):>7}%  {d_a:>+7.2f}%  {s_a}')
        print(f'  {"Test Loss":<12} {test_loss:>7.2f}%  '
              f'{tgt.get("test_loss","--"):>7}%  {d_l:>+7.2f}%  {s_l}')
        print(sep)

        verdict = '✅ DENTRO da tolerância (±2%)' if (abs(d_a) <= TOLERANCE and abs(d_l) <= TOLERANCE * 3) \
                  else '⚠️  FORA da tolerância — revisar hiperparâmetros'
        print(f'  {verdict}\n')

    def close(self):
        for f in self._files.values():
            f.close()
