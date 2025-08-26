# Hotel Review Sentiment Classification Project

## Overview
This project aims to classify hotel reviews from Tripadvisor into three sentiment categories:
- **Negative** (1–2 stars)
- **Neutral** (3 stars)
- **Positive** (4–5 stars)

Using deep learning models with **pre-trained GloVe embeddings**, we built and evaluated four neural network architectures: Dense, CNN, LSTM, and BiLSTM + Attention.

All models were trained both **with and without class weights** to handle class imbalance.

---

## Dataset
- **Source**: `tripadvisor_hotel_reviews.csv`
- **Size**: 20492 reviews
- **Class Distribution**:
  - Positive: ~74% (3,019)
  - Negative: ~16% (643)
  - Neutral: ~10% (437)

> ⚠️ **Challenge**: The dataset is imbalanced — Neutral class is underrepresented.

---

## Text Preprocessing
- Converted text to lowercase
- Removed HTML, URLs, punctuation
- Removed stopwords using NLTK
- Applied lemmatization
- Tokenized and padded sequences (max length: 150)

---

## Embedding
- Used **GloVe 6B.100d** pre-trained word vectors
- Built a custom embedding matrix of shape `(5000, 100)`
- Kept embedding layer **non-trainable** to preserve semantic meaning

---

## Models Built

| Model | Key Features |
|------|-------------|
| **Dense** | GlobalAveragePooling + 2 Dense layers |
| **CNN** | Conv1D(128,5) + GlobalMaxPooling1D |
| **LSTM** | Single LSTM layer with dropout |
| **BiLSTM+Attention** | Bidirectional LSTM + Custom Attention Layer |

All models used:
- Adam optimizer
- Categorical crossentropy loss
- EarlyStopping & ReduceLROnPlateau
- Dropout (0.5–0.6), L2 regularization (CNN)

---

## Training Strategy
- Trained each model **twice**:
  1. Without class weights
  2. With `class_weight='balanced'`
- Train/Test split: 80/20, stratified by sentiment
- Batch size: 128, Max epochs: 20 (with early stopping)

---

## Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score (per class)
- Confusion Matrix
- Focus on **F1-score (weighted)** and **Neutral recall**

---

## Key Findings

### 1. **Class Imbalance is a Major Challenge**
- All models initially **ignored the Neutral class**
- Without class weights, **Neutral recall was near 0%**

### 2. **Class Weights Improved Nuance**
- After applying `class_weight`, models became more sensitive to mixed-sentiment reviews
- Example:  
  `"The hotel was great, but the AC didn’t work."` → now predicted as **Neutral** by some models

### 3. **CNN Performed Best Overall**
- Highest **F1-score (0.82)** and **validation accuracy (84.4%)**
- Fast training (~10s/epoch)
- Robust to overfitting

### 4. **BiLSTM+Attention Did Not Outperform CNN**
- Despite complexity, it didn’t significantly improve on Neutral recall
- Very slow training (~90s/epoch)
- May need attention visualization to debug

---

## Best Model
✅ **CNN trained with class weights**

Reasons:
- High accuracy and F1-score
- Fast inference
- Handles mixed sentiment better than baseline
- Simpler and more reliable than RNNs

---

## Future Work
- Generate synthetic Neutral reviews
- Visualize attention weights
- Use LIME/SHAP for explainability
- Deploy as API or web app

---