# Arabic Sentiment Analysis

A deep learning project that fine-tunes a DistilBERT model to classify Arabic text reviews into sentiment categories (Positive, Negative, Mixed).

## Overview

This project uses transfer learning with a pre-trained multilingual DistilBERT model from TensorFlow Hub to perform sentiment analysis on Arabic reviews. The model is trained on a dataset of 100,000 Arabic reviews and deployed on Google Cloud Vertex AI for inference.

## Features

- Fine-tuned DistilBERT model for Arabic text classification
- Three-class sentiment classification: Positive, Negative, and Mixed
- Streamlit web interface for real-time predictions
- Dockerized training pipeline for Google Cloud Vertex AI
- Integration with Google Cloud AI Platform for model deployment

## Project Structure

```
.
├── app.py                              # Streamlit web application
├── classifier/
│   ├── data/
│   │   └── ar_reviews_100k.tsv        # Training dataset (100k Arabic reviews)
│   ├── trainer/
│   │   ├── model.py                   # Model architecture and training logic
│   │   ├── task.py                    # Training task entry point with CLI args
│   │   └── __init__.py
│   ├── Dockerfile                      # Docker container for training
│   └── SavedModel/                     # Exported trained model
├── classify_text_with_distilbert.ipynb # Jupyter notebook for experimentation
├── emotions/                           # UI assets (happy/sad images)
└── README.md
```

## Model Architecture

The model consists of:
1. **Preprocessing Layer**: DistilBERT tokenizer and text preprocessing
2. **Encoder Layer**: Pre-trained DistilBERT (6-layer, 768-hidden, 12-heads)
3. **Dropout Layer**: Regularization (default rate: 0.15)
4. **Dense Output Layer**: 3-class softmax classifier

Total parameters: ~65M (all trainable during fine-tuning)

## Dataset

- **Source**: Arabic reviews dataset (`ar_reviews_100k.tsv`)
- **Size**: 100,000 reviews
- **Labels**: Positive, Negative, Mixed
- **Split**: 80% training, 20% testing
- **Encoding**: One-hot encoded labels

## Training

### Local Training

```bash
python -m classifier.trainer.task \
  --train_data_path=./classifier/data/ar_reviews_100k.tsv \
  --output_dir=./classifier/SavedModel \
  --batch_size=32 \
  --epochs=5 \
  --lr=0.001 \
  --dropout_rate=0.15
```

### Docker Training (for Vertex AI)

```bash
cd classifier
docker build -t arabic-sentiment-trainer .
docker run arabic-sentiment-trainer \
  --train_data_path=/code/data/ar_reviews_100k.tsv \
  --output_dir=/code/SavedModel \
  --epochs=5
```

## Deployment

The model is deployed on Google Cloud Vertex AI:
- **Project ID**: 1033908600341
- **Endpoint ID**: 1064408619547623424
- **Region**: us-central1

## Web Application

Run the Streamlit app locally:

```bash
streamlit run app.py
```

The app provides:
- Text input for Arabic reviews
- Real-time sentiment prediction
- Visual feedback with emotion images (happy/sad)

## Requirements

### Training
- TensorFlow 2.12+
- TensorFlow Hub
- TensorFlow Text
- tf-models-official 2.13.2
- pandas
- numpy
- scikit-learn

### Inference
- streamlit
- google-cloud-aiplatform
- numpy

## Usage Example

```python
from google.cloud import aiplatform

# Initialize prediction
predictions = predict_custom_trained_model_sample(
    project="1033908600341",
    endpoint_id="1064408619547623424",
    location="us-central1",
    instances={"text": "هذا الفندق رائع جداً"}
)

# Output: Positive sentiment
```

## Model Performance

The model uses:
- **Optimizer**: AdamW with warmup
- **Loss**: Categorical Cross-Entropy
- **Metrics**: Categorical Accuracy
- **Validation Split**: 20%

## License

This project is for educational and research purposes.

## Acknowledgments

- DistilBERT model from [Kaggle Models](https://www.kaggle.com/models/jeongukjae/distilbert)
- TensorFlow Hub for model hosting
- Google Cloud Vertex AI for deployment infrastructure
