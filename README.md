# Análise Comparativa de Arquiteturas de CNNs com Técnicas de Pré-processamento para Classificação de Câncer Renal em Imagens Tomográficas

Este repositório apresenta a implementação da **Etapa 1** do projeto de **Iniciação Científica (PIBIC)**, orientado pelo **Prof. Dr. Wheidima Carneiro de Melo** na **Escola Superior de Tecnologia (EST/UEA)**.

O trabalho investiga o impacto de técnicas de pré-processamento de contraste — **CLAHE** e **correção gama** (γ = 0,7) — comparadas à condição base, sobre o desempenho de cinco arquiteturas de CNNs (**ResNet-18, ResNet-34, ResNet-50, AlexNet e VGG-16**) na classificação binária (Normal vs. Tumor) de imagens de tomografia computadorizada renais. O artigo derivado deste trabalho foi submetido ao **XXX Congresso Brasileiro de Engenharia Biomédica (CBEB 2026)**.

---

## 📁 Estrutura do Repositório

```text
kidney-ct-cnn-classification/
│
├── data/
│   └── kidneyData.csv               # Metadados do CT Kidney Dataset (Kaggle)
│
├── src/
│   ├── dataset.py                   # Split treino/val/teste e pré-processamento (Base, CLAHE, Gama)
│   ├── model_resnet18.py            # Definição do modelo ResNet-18 (transfer learning)
│   ├── model_resnet34.py            # Definição do modelo ResNet-34 (transfer learning)
│   ├── model_resnet50.py            # Definição do modelo ResNet-50 (transfer learning)
│   ├── model_alexnet.py             # Definição do modelo AlexNet (transfer learning)
│   ├── model_vgg16.py               # Definição do modelo VGG-16 (transfer learning)
│   ├── model_mobilenetv2.py         # Definição do modelo MobileNetV2 (experimento adicional)
│   ├── model_efficientnetb0.py      # Definição do modelo EfficientNet-B0 (experimento adicional)
│   ├── trainer.py                   # Loop de treino, validação e avaliação em teste
│   ├── logger.py                    # Registro de métricas por época (CSV)
│   └── targets.py                   # Metas de acurácia do artigo original (RITA) para validação
│
├── results/
│   └── *_log.csv                    # Logs de treino por modelo/condição
│
├── gradcam_output/
│   └── gradcam_*.png                # Mapas de ativação (Grad-CAM) por modelo/condição
│
├── docs/
│   ├── main.tex                     # Artigo em LaTeX (formato IEEEtran, submetido ao CBEB 2026)
│   ├── fig_acc_curves.png           # Figura: curvas de acurácia de treino/validação
│   ├── fig_loss_curves.png          # Figura: curvas de perda de treino/validação
│   ├── fig_confusion.png            # Figura: matrizes de confusão dos 15 experimentos
│   ├── sample_normal.jpg            # Amostra de imagem CT — classe Normal
│   ├── sample_tumor.jpg             # Amostra de imagem CT — classe Tumor
│   ├── metricas_avaliacao.csv       # Métricas de referência do artigo original (RITA)
│   ├── metricas_artigo.txt          # Metas de acurácia por modelo (artigo original)
│   └── plot_results.py              # Script de geração das figuras de resultados
│
├── train.py                         # Script principal de treinamento (seleção interativa de experimentos)
├── collect_metrics.py               # Consolidação das métricas de teste dos experimentos
├── retest.py                        # Reavaliação de pesos salvos no conjunto de teste
├── summary.py                       # Geração de resumo comparativo dos resultados
├── gradcam.py                       # Geração dos mapas de ativação Grad-CAM
├── .gitignore
└── README.md                        # Documentação do repositório
```

> **Nota:** as pastas `runs/` (pesos `.pt` e imagens de batch geradas durante o treino) não são versionadas por conterem artefatos binários grandes e reprodutíveis a partir do código.

---

## Objetivo do Projeto

Avaliar, de forma sistemática, se técnicas simples de realce de contraste (CLAHE e correção gama) melhoram a capacidade de arquiteturas de CNNs pré-treinadas em diferenciar imagens de TC renal **Normais** de imagens com **Tumor**, servindo como etapa preliminar para o diagnóstico assistido por imagem do câncer renal.

### Classes do dataset

- `0: Normal`
- `1: Tumor`

### Condições de pré-processamento avaliadas

- **Base** — imagem original, sem pré-processamento
- **CLAHE** — equalização adaptativa de histograma com limitação de contraste
- **Gama** — correção gama com fator γ = 0,7

---

## Fluxo de Trabalho

### 1. Preparação dos Dados

O `CT Kidney Dataset` (Kaggle, 12.446 imagens) é filtrado para as classes Normal e Tumor (2.283 imagens/classe, 4.566 total) e dividido em **treino (60%) / validação (20%) / teste (20%)**, com semente fixa (`seed=42`) para reprodutibilidade. O download e a organização são feitos automaticamente via `kagglehub` dentro de `src/dataset.py`, chamado por `train.py`.

### 2. Treinamento dos Modelos

O script `train.py` oferece seleção interativa de modelos e condições a treinar. Para cada combinação, é aplicado **transfer learning**: o *backbone* convolucional é congelado (pesos pré-treinados na ImageNet) e apenas o classificador final é treinado para a tarefa binária.

Os resultados de cada execução (pesos, curvas, matrizes de confusão, métricas) são salvos em `runs/<modelo>_<condicao>/`.

### 3. Avaliação

- `collect_metrics.py` consolida as métricas de teste (acurácia, precisão, recall, F1) de todos os experimentos.
- `retest.py` permite reavaliar pesos já salvos no conjunto de teste, sem retreinar.
- `summary.py` gera um resumo comparativo final.

### 4. Interpretabilidade

`gradcam.py` gera mapas de ativação (Grad-CAM) para cada modelo/condição, evidenciando as regiões da imagem mais relevantes para a decisão da rede — salvos em `gradcam_output/`.

---

## Tecnologias e Bibliotecas

- Python 3.10
- PyTorch + TorchVision
- OpenCV (pré-processamento CLAHE/gama)
- Scikit-learn (métricas de avaliação)
- Pandas / NumPy
- Matplotlib (visualizações e curvas)
- KaggleHub (download do dataset)

---

## Como Executar

### Treinar os modelos

```bash
python train.py
```

> O script solicita interativamente quais modelos e condições de pré-processamento treinar.

### Consolidar métricas dos experimentos

```bash
python collect_metrics.py
```

### Gerar mapas Grad-CAM

```bash
python gradcam.py
```

> Certifique-se de que o ambiente possui PyTorch com suporte a CUDA e que a biblioteca `kagglehub` está configurada para o download do dataset.

---

## Resultados

Métricas no conjunto de teste (914 amostras), em percentual:

| Modelo | Condição | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|---|
| ResNet-18 | Base | 96,94 | 97,35 | 96,50 | 96,92 |
| ResNet-18 | CLAHE | 95,19 | 95,38 | 94,97 | 95,18 |
| ResNet-18 | Gama | **97,48** | 97,17 | **97,81** | **97,49** |
| ResNet-34 | Base | 95,19 | 95,59 | 94,75 | 95,16 |
| ResNet-34 | CLAHE | 95,40 | 95,21 | **95,62** | 95,41 |
| ResNet-34 | Gama | **95,73** | **97,07** | 94,31 | **95,67** |
| ResNet-50 | Base | 95,62 | 95,42 | 95,84 | 95,63 |
| ResNet-50 | CLAHE | 96,06 | **96,67** | 95,40 | 96,04 |
| ResNet-50 | Gama | **96,50** | 96,50 | **96,50** | **96,50** |
| AlexNet | Base | **98,91** | 98,48 | **99,34** | **98,91** |
| AlexNet | CLAHE | 97,16 | 96,95 | 97,37 | 97,16 |
| AlexNet | Gama | 98,80 | 98,48 | 99,12 | 98,80 |
| VGG-16 | Base | 98,91 | 98,06 | 99,78 | 98,92 |
| VGG-16 | CLAHE | 98,36 | **99,11** | 97,59 | 98,35 |
| VGG-16 | Gama | **99,12** | 98,28 | **100,00** | **99,13** |

O melhor resultado global foi obtido pela **VGG-16 com correção gama**, alcançando **99,12% de acurácia** e **100% de recall** na classe Tumor. De modo geral, a correção gama trouxe ganhos consistentes de 0,21 a 0,88 p.p. em relação à condição base na maioria dos modelos.

Análise detalhada, discussão e referências bibliográficas completas estão disponíveis no artigo em [`docs/main.tex`](docs/main.tex).

---

## Observações

- O dataset original (`CT Kidney Dataset`) é público e está disponível no [Kaggle](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone).
- Este repositório contempla apenas as classes **Normal** e **Tumor**; o dataset original também possui as classes Cyst e Stone, não utilizadas neste trabalho.
- Os pesos treinados (`.pt`) não são versionados neste repositório; podem ser reproduzidos executando `train.py` com o mesmo `seed=42`.
- Os experimentos com MobileNetV2 e EfficientNet-B0 (`src/model_mobilenetv2.py`, `src/model_efficientnetb0.py`) fazem parte da exploração inicial do projeto e não integram o conjunto de resultados reportado no artigo.

---

## Autores

| [<img src="https://github.com/thiagocordeirum.png?size=100" width=100><br><sub>Thiago Cordeiro de Melo</sub>](https://github.com/thiagocordeirum) |
|:---:|

---

## Orientação

- **Prof. Dr. Wheidima Carneiro de Melo**
- **Escola Superior de Tecnologia – Universidade do Estado do Amazonas (EST/UEA)**
- **Programa Institucional de Bolsas de Iniciação Científica (PIBIC)**
