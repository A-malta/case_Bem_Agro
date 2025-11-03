# Desafio IA — Segmentação de Vegetação

Este projeto realiza a segmentação de imagens aéreas (ortomosaicos) para detecção de vegetação, utilizando técnicas de visão computacional e aprendizado de máquina.

## Estrutura do Projeto

## Requisitos

Instale as dependências necessárias:

```bash
pip install -r requirements.txt
```

Bibliotecas:
```
rasterio
opencv-python
numpy
tensorflow
tqdm       
matplotlib   
scikit-learn
```

## Execução

### 1. Divisão do Ortomosaico

```bash
python divide_orthomosaic.py --input orthomosaic.tif --output dataset/crops
```

### 2. Binarização das Imagens

```bash
python binarize_images.py --input dataset/crops --output dataset/masks
```

### 3. Treinamento do Modelo

```bash
python train_model.py --rgb dataset/crops --groundtruth dataset/masks --modelpath model/model.h5
```

### 4. Inferência com o Modelo Treinado

```bash
python model_inference.py --rgb dataset/images_validation --modelpath model --output results
```

## Resultados

<p align="center">
  <img src="results/1.jpg" alt="Resultado 1" width="100%" style="margin-bottom:20px;">
  <img src="results/2.jpeg" alt="Resultado 2" width="100%" style="margin-bottom:20px;">
  <img src="results/3.jpg" alt="Resultado 3" width="100%" style="margin-bottom:20px;">
  <img src="results/4.jpeg" alt="Resultado 4" width="100%" style="margin-bottom:20px;">
  <img src="results/5.jpeg" alt="Resultado 5" width="100%" style="margin-bottom:20px;">
  <img src="results/6.jpeg" alt="Resultado 6" width="100%" style="margin-bottom:20px;">
  <img src="results/7.jpeg" alt="Resultado 7" width="100%" style="margin-bottom:20px;">
</p>
