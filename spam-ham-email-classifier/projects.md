# Machine Learning & NLP Projects

This document outlines a set of end-to-end machine learning projects, framed as **portfolio-ready projects** with clear objectives, datasets, models, evaluation metrics, and deployment components.

---

## 1. Spam vs Ham Email Classifier

**Problem**
Classify emails as **spam** or **ham** using a fine-tuned transformer model on a real-world email dataset.

**Dataset**

* Enron Spam Email Dataset

**Approach**

* Text preprocessing and cleaning
* Tokenization using Hugging Face tokenizers
* Fine-tuning a pretrained transformer
* Binary classification

**Models**

* BERT
* DistilBERT

**Evaluation Metrics**

* Accuracy
* Precision
* Recall
* F1-score

**Tech Stack**

* Python
* PyTorch
* Hugging Face Transformers & Datasets
* scikit-learn
* Streamlit (interactive UI for predictions)

**Deliverables**

* Trained model checkpoint
* Streamlit web app for live email classification
* Training & evaluation notebook/scripts

---

## 2. Sentiment Analysis on Movie Reviews

**Problem**
Predict whether a movie review expresses **positive** or **negative** sentiment.

**Dataset**

* IMDB Movie Reviews Dataset

**Approach**

* Binary text classification
* Fine-tuning pretrained language models
* Train/validation/test split

**Models**

* DistilBERT
* RoBERTa

**Evaluation Metrics**

* Accuracy
* F1-score

**Tech Stack**

* Python
* PyTorch / TensorFlow
* Hugging Face Transformers
* NumPy

**Deliverables**

* Fine-tuned sentiment model
* Evaluation report
* Inference script for batch predictions

---

## 3. News Article Topic Classification

**Problem**
Classify news articles into multiple topics such as **Sports**, **Politics**, **Technology**, and **Business**.

**Dataset**

* AG News Dataset

**Approach**

* Multi-class text classification
* Label encoding
* Transformer-based text representations

**Models**

* BERT

**Evaluation Metrics**

* Accuracy
* Per-class Precision & Recall

**Tech Stack**

* Python
* PyTorch
* Hugging Face Transformers
* Pandas
* Matplotlib
* Seaborn

**Deliverables**

* Trained topic classifier
* Confusion matrix visualization
* Performance plots

---

## 4. Product Review Rating Classification (1–5 Stars)

**Problem**
Predict product review ratings from **1 to 5 stars** based on review text, handling class imbalance.

**Dataset**

* Amazon Reviews Dataset

**Approach**

* Multi-class text classification
* Class imbalance handling (weighted loss / sampling)
* Transformer fine-tuning

**Models**

* BERT
* DeBERTa

**Evaluation Metrics**

* Macro F1-score
* Confusion Matrix

**Tech Stack**

* Python
* PyTorch
* Hugging Face Transformers & Datasets
* scikit-learn

**Deliverables**

* Rating prediction model
* Confusion matrix visualization
* Evaluation report

---

## 5. Resume Category Classifier

**Problem**
Automatically classify resumes into predefined **job roles** and expose predictions via an API.

**Dataset**

* Kaggle Resume Dataset

**Approach**

* Text preprocessing with NLP tools
* Multi-class classification
* Model serving via REST API

**Models**

* DistilBERT

**Evaluation Metrics**

* Accuracy
* F1-score

**Tech Stack**

* Python
* PyTorch
* Hugging Face Transformers
* spaCy
* FastAPI

**Deliverables**

* Resume classification model
* FastAPI service for inference
* API documentation

---

## 6. Toxic Comment Classification

**Problem**
Detect **toxic and abusive content** in online comments using multi-label classification.

**Dataset**

* Jigsaw Toxic Comment Classification Dataset

**Approach**

* Multi-label text classification
* Sigmoid outputs for multiple toxicity labels
* Threshold tuning

**Models**

* RoBERTa

**Evaluation Metrics**

* Precision-Recall
* F1-score per label

**Tech Stack**

* Python
* PyTorch
* Hugging Face Transformers
* Pandas

**Deliverables**

* Toxicity detection model
* Precision-Recall curves
* Inference pipeline for moderation use cases

---

## Notes

* All projects are suitable for **portfolio, interviews, and production demos**.
* Each project can be extended with experiment tracking, hyperparameter tuning, and deployment.
* Models can be swapped or upgraded with minimal pipeline changes.
