# Text Classification with Pre-trained BERT and Fine-tuned DistilBERT Models

This project demonstrates a **Streamlit-based web application** that classifies text using two models: a **pre-trained BERT model** and a **fine-tuned DistilBERT model**. The app allows users to input text, view predictions from both models, and visualize the confidence levels of predictions through bar charts.

## Features

- **Pre-trained BERT Model**: A BERT model trained on a general text corpus to predict one of five news categories: `entertainment`, `business`, `sport`, `politics`, or `tech`.
- **Fine-tuned DistilBERT Model**: A DistilBERT model fine-tuned on a custom dataset to improve accuracy for the same classification task.
- **Streamlit Web App**: A simple, interactive web app for user input and model prediction. It displays prediction results from both models and plots confidence levels.

## Demo

1. Enter text in the input box for classification.
2. View predictions from:
   - **Pre-trained BERT model**
   - **Fine-tuned DistilBERT model**
3. Confidence levels are displayed in bar charts for both models.


pip install transformers torch streamlit pandas matplotlib
![image](https://github.com/user-attachments/assets/865ce131-2f1d-448f-88d4-b0b751e16485)
![image](https://github.com/user-attachments/assets/a2b36d6d-f046-4ee9-b77a-d813bca8eefb)
![image](https://github.com/user-attachments/assets/3ddec5f4-8ac8-4457-a0ef-f949fa2cc168)

