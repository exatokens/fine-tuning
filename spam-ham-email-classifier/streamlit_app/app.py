import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "./enron-spam-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

st.title("Spam vs Ham Email Classifier")

email_text = st.text_area("Paste email text here:")

if st.button("Classify") and email_text.strip():
    inputs = tokenizer(email_text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
    ham_prob = float(probs[0])
    spam_prob = float(probs[1])

    st.write(f"**Prediction:** {'Spam' if spam_prob > ham_prob else 'Ham'}")
    st.write(f"Ham: {ham_prob:.3f}, Spam: {spam_prob:.3f}")

