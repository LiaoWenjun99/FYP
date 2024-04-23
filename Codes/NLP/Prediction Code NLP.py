# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 21:14:12 2024

@author: Wenjun
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
# Load the saved tokenizer and model
output_dir = "trained_model_test2_100000sample"

tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = AutoModelForSequenceClassification.from_pretrained(output_dir)

# FNS_dataframe = pd.read_csv('FNS Sequences for ML model validation.csv')
FNS_dataframe = pd.read_csv('Sangeetha_Seq_for_verification.csv')

FNS_dataframe = FNS_dataframe.dropna()
FNS_samples = FNS_dataframe['Sample']
FNS_samples.reset_index(drop=True, inplace=True)
FNS_sequence = FNS_dataframe['DNA Sequence']
FNS_sequence.reset_index(drop=True, inplace=True)

# Extract sequences from the DataFrame
new_sequences = FNS_sequence.tolist()

# Tokenize the new sequences
tokenized_sequences = tokenizer(new_sequences, return_tensors="pt", padding=True, truncation=True)

# Perform inference
model.eval()
with torch.no_grad():
    outputs = model(**tokenized_sequences)

# Get the predicted labels
predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()

# Add the predicted labels to the DataFrame
FNS_dataframe['Predicted_Labels'] = predicted_labels

# Save the DataFrame with predicted labels
FNS_dataframe.to_csv(output_dir + 'Sangeetha.csv', index=False)
