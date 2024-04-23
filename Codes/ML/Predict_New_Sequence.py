# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 22:39:16 2023

@author: Wenjun
"""

# Goal: To predict a new unseen DNA sequence if the efficiency is High, Medium, or Low.
# Steps: 1)Load in Trained Model, 2) Load in new Sequence. 
# 3) Convert new seq into One Hot encoding that has the same foramt as trained model
# 4) Predict OHE version using trained model

# In[1]:
import sys
import pandas as pd
import pickle
import numpy as np
import time
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


# Load in Trained Model
RFC = pickle.load(open('RFC_Feature_Selection_100.pkl', 'rb'))
eg = pd.read_csv('RFC_Feature_Selection_100_cols_OHE.csv',index_col= None)

# Load in new sequence
# Seq = 'TTAGTGAAATGACGCGTCTTTCCCGCGAGTTCTTCGCTCTGCCAGCCGAAGAGAAACTTGAATACGATACCACTGGCGGAAAACGTGGCGGTTTTACAATCTCAACCGTGCTGCAAGGAGACGATGCTATGGATTGGCGCGAGTTCGTTACTTATTTCTCTTACCCAATTAACGCTCGCGTTTACTCGCGTTGGCCTAAGAAACCCGTGGGGGGGCGCTCAACAACAGAGGGTTAGAGGGGATAGTGAATGGTCCTTGGTGCGAAGCTTCTGGAGGTTTTAAGTGAAGCGATGGGTTTGGAAAAGGGCGACCTTACTAAAGCCTGTGTAGACATGGAACAAAAAGTTTTAATCAACTACTACCCTACCTGTCCACAGCCTGACTTGACATTGGGGGTACGCCGCCATACTGACCCTGGGACAATTACGATTTTATTACAAGACATGGTTGGGGGTTTGCAAGCAACTCGTGATGGTGGAAAAACGTGGATCACGGTCCAA'

# # Convert new seq into df
# char_list = list(Seq)

# # Create a DataFrame with a single row and columns corresponding to the length of the string
# df = pd.DataFrame([char_list], columns=range(1, len(Seq) + 1))
# X_input = df_OHE.loc[:,df_OHE.columns != 'class'] 
# df_OHE = pd.get_dummies(df).reindex(columns=eg.columns,fill_value=0)
FNS_dataframe = pd.read_csv('FNS Sequences for ML model validation.csv')
FNS_dataframe = FNS_dataframe.dropna()

# Selecting column '0' and splitting it into columns 
FNS_samples = FNS_dataframe['Sample']
FNS_samples.reset_index(drop=True, inplace=True)
FNS_dataframe_processed = FNS_dataframe['DNA Sequence'].str.split('',expand=True)
# Removing the first col of 0 followed by last col
FNS_dataframe_processed = FNS_dataframe_processed.iloc[:,1:]
FNS_dataframe_processed.drop(columns=FNS_dataframe_processed.columns[-1],axis=1,inplace=True)
FNS_dataframe_processed.drop(columns=FNS_dataframe_processed.columns[-1],axis=1,inplace=True)


# One Hot Encoding #

df_OHE = pd.get_dummies(FNS_dataframe_processed).reindex(columns=eg.columns,fill_value=0)
X_input = df_OHE

Y_pred = RFC.predict(X_input)
print(Y_pred)

class_mapping = {0: 'High', 1: 'Medium', 2: 'Low'}
# class_mapping = {0: 'High', 1:'Low'}


for key, value in class_mapping.items():
    print(f'{key}: {value}')
    
# Ensure that the keys in Y_pred are present in class_mapping
predicted_labels = [class_mapping.get(idx, 'Unknown') for idx in Y_pred]

# 'Unknown' will be used for any key in Y_pred that is not present in class_mapping

for key, value in class_mapping.items():
    print(f'{key}: {value}')
predicted_labels_series = pd.Series(predicted_labels, name='Predicted_Labels')

new_data_with_predictions = pd.concat([FNS_samples, predicted_labels_series], axis=1)

new_data_with_predictions.to_csv('Prediction_RFC_feature_sel_100.txt', sep='\t', index=False)


