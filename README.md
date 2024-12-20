# LaTeX OCR: Hand-written Math Formulas to LaTeX

## Introduction

This is the project for the course "Neuroscience" (INT3421 55).

## Team member

Nguyen Quang Huy - 22028077 (leader)

Nguyen Thanh Dao - 22028113

Nguyen Phuong Anh - 22028332

## Project structure

```
LaTex OCR
├── latex-ocr
|   ├── training-data
|   |   └── 0.jpg
|   |   └── 1.jpg
|   |   └── ...
|   ├── test-data
|   |   └── 0.jpg
|   |   └── 1.jpg
|   |   └── ...
|   ├── training-data-label.json
|   └── test-data-label.json
├── models
|   ├── latexocr-finetuned
|   |   └── ...
├── datasets.py
├── inference.py
├── train.py
├── utils.py
└── README.md
```

## Installation

Download the dataset in this [link](https://www.kaggle.com/datasets/staticpunch/dataset-100k-images-trocr). Add the dataset into the folder and make sure to set the name correctly following the given structure.

Train the model by running the following line:

```
python train.py
```

After training the model, infer the result by running the following line:
```
python inference.py --img_url={image-url}
```
