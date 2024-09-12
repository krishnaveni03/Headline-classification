Text Classification with Pre-trained BERT and Fine-tuned DistilBERT Models
This project demonstrates a Streamlit-based web application that classifies text using two models: a pre-trained BERT model and a fine-tuned DistilBERT model. The app allows users to input text, view predictions from both models, and visualize the confidence levels of predictions through bar charts.

Features
Pre-trained BERT Model: A BERT model trained on a general text corpus to predict one of five news categories: entertainment, business, sport, politics, or tech.
Fine-tuned DistilBERT Model: A DistilBERT model fine-tuned on a custom dataset to improve accuracy for the same classification task.
Streamlit Web App: A simple, interactive web app for user input and model prediction. It displays prediction results from both models and plots confidence levels.
Demo
Enter text in the input box for classification.
View predictions from:
Pre-trained BERT model.
Fine-tuned DistilBERT model.
Confidence levels are displayed in bar charts for both models.
Setup Instructions
Prerequisites
Python 3.7+
Install the required packages:
bash
Copy code
pip install transformers torch streamlit pandas matplotlib
Project Structure
bash
Copy code
|-- app.py                # Streamlit app script
|-- fine_tuned_model/      # Directory containing the fine-tuned DistilBERT model
|-- pretrained_model/      # Directory containing the pre-trained BERT model
|-- README.md              # This file
Models
Pre-trained BERT model is loaded from bert-base-uncased using Hugging Face's Transformers library.
Fine-tuned DistilBERT model is a custom model trained on a news dataset.
Dataset
The model has been fine-tuned on a dataset containing text and labels for 5 categories:

entertainment
business
sport
politics
tech
Running the App
Clone this repository and navigate to the project directory.

Run the Streamlit app:

bash
Copy code
streamlit run app.py
The app will open in your default browser. Enter text into the input field and view the classification results.
Example Output
For a given input text, the app will output:

Predicted label by the pre-trained BERT model.
Predicted label by the fine-tuned DistilBERT model.
Confidence levels for both models, displayed as bar charts.
Fine-Tuning Process
The DistilBERT model was fine-tuned using Hugging Face's Trainer on a custom news classification dataset. The labels used were converted to categorical values for training. The fine-tuned model is saved under the fine_tuned_model/ directory.
