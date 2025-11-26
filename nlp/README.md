# Mountain Named Entity Recognition

BERT-based NER model for identifying mountain names in text.

## Dataset

- Size: 1584 samples
- Format: CSV with text and character markers
- Tagging: BIO scheme (B-MOUNTAIN, I-MOUNTAIN, O)
- Dataset link: [Kaggle](https://www.kaggle.com/datasets/geraygench/mountain-ner-dataset?select=mountain_dataset_with_markup.csv)

## Model

- Architecture: BERT-base-cased fine-tuned for token classification
- Labels: O, B-MOUNTAIN, I-MOUNTAIN
- Model weights: [Google Drive](https://drive.google.com/drive/folders/1wklMdYWQcn6WGiToUG2R7IuboWVvSeh4?usp=drive_link)

## Performance

| Metric    | Score  |
| --------- | ------ |
| Precision | 0.9184 |
| Recall    | 0.9783 |
| F1-score  | 0.9474 |

## Project Structure

```
ner/
 data/
    mountain_dataset.csv
 mountain_ner_model/
 train.py
 inference.py
 dataset_preparation.ipynb
 requirements.txt
 README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Dataset Preparation

See `dataset_preparation.ipynb` for the complete dataset creation process:

- Loading CSV with character markers
- Converting to BIO tags
- Train/val split
- Tokenization examples

### Training

```bash
python train.py --dataset_path data/mountain_dataset.csv --output_dir mountain_ner_model
```

### Inference

Single text:

```bash
python inference.py --model_path mountain_ner_model --text "I climbed Mount Everest"
```

From file:

```bash
python inference.py --model_path mountain_ner_model --input_file texts.txt
```
