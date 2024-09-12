import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertForSequenceClassification
import torch
import pandas as pd

fine_tuned_model_path = 'fine_tuned_model'  


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

fine_tuned_model = DistilBertForSequenceClassification.from_pretrained(fine_tuned_model_path, num_labels=5)
fine_tuned_tokenizer = BertTokenizer.from_pretrained(fine_tuned_model_path)


labels = ['entertainment', 'business', 'sport', 'politics', 'tech']


def predict(text, model, tokenizer, is_distilbert=False):
    encodings = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')

    if is_distilbert:
        encodings.pop('token_type_ids', None)
    
    with torch.no_grad():
        outputs = model(**encodings)
    logits = outputs.logits
    predictions = torch.softmax(logits, dim=-1).squeeze().tolist()
    prediction = torch.argmax(logits, dim=-1).item()
    return labels[prediction], predictions


st.title('Text Classification with Pre-trained BERT and Fine-tuned Model')

text_input = st.text_area("Enter text for classification:")

if text_input:
    
    pretrain_prediction, pretrain_confidences = predict(text_input, bert_model, bert_tokenizer)
    
    
    finetune_prediction, finetune_confidences = predict(text_input, fine_tuned_model, fine_tuned_tokenizer, is_distilbert=True)
    
    
    st.write("### Pre-trained BERT Model Prediction:")
    st.write(f"**Predicted Label:** {pretrain_prediction}")
    
    st.write("### Fine-tuned Model Prediction:")
    st.write(f"**Predicted Label:** {finetune_prediction}")

    
    pretrain_conf_df = pd.DataFrame({
        'Label': labels,
        'Confidence': pretrain_confidences
    })
    
    finetune_conf_df = pd.DataFrame({
        'Label': labels,
        'Confidence': finetune_confidences
    })

    
    st.write("### Confidence Levels for Pre-trained BERT Model:")
    st.bar_chart(pretrain_conf_df.set_index('Label'))

    st.write("### Confidence Levels for Fine-tuned Model:")
    st.bar_chart(finetune_conf_df.set_index('Label'))
