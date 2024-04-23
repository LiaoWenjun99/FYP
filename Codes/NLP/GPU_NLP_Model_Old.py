#!/usr/bin/env python
# coding: utf-8
 
# In[1]:


import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification

from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset
from accelerate import Accelerator
import logging
from sklearn.metrics import precision_recall_fscore_support


# In[2]:


# Cell 2: Define the working device
device = torch.device("cuda")


# In[3]:


# Cell 3: Load the model
num_class = 3
model = AutoModelForSequenceClassification.from_pretrained("/hpctmp/e0543831/GPU/InstaDeepAI", num_labels=num_class)
model = model.to(device)


# In[4]:


# Cell 4: Import dataset
samples_dataframe_H = pd.read_csv("data_H.csv", names=['sequence'])
samples_dataframe_M = pd.read_csv("data_M.csv", names=['sequence'])
samples_dataframe_L = pd.read_csv("data_L.csv", names=['sequence'])
#samples_dataframe_H = pd.read_csv("unique_sequences_H.csv", names=['sequence'])
#samples_dataframe_M = pd.read_csv("unique_sequences_M.csv", names=['sequence'])
#samples_dataframe_L = pd.read_csv("unique_sequences_L.csv", names=['sequence'])


# In[5]:


# Cell 5: Data Preprocessing
num_random_rows = 400000
H_rand = samples_dataframe_H.sample(n=num_random_rows)
H_rand = H_rand[~H_rand['sequence'].str.contains('N')]

M_rand = samples_dataframe_M.sample(n=num_random_rows)
M_rand = M_rand[~M_rand['sequence'].str.contains('N')]

L_rand = samples_dataframe_L.sample(n=num_random_rows)
L_rand = L_rand[~L_rand['sequence'].str.contains('N')]

samples_all = pd.concat([H_rand, M_rand, L_rand])


# In[6]:


# Cell 6: Creating labels for H, M, L
H_perflist_H = [0]*len(H_rand)
H_perflist_M = [1]*len(M_rand)
H_perflist_L = [2]*len(L_rand)
H_perflist = H_perflist_H + H_perflist_M + H_perflist_L

H_perflist = np.array(H_perflist)
df_temp = pd.DataFrame(H_perflist, columns=['label'], dtype='int32', index=samples_all.index)
samples_all = pd.concat([samples_all, df_temp], axis=1)


# In[7]:


# Cell 7: Split the combined dataset into training, validation, and test sets
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

train_data, val_test_data = train_test_split(samples_all, test_size=val_ratio + test_ratio, random_state=42)
val_data, test_data = train_test_split(val_test_data, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

train_sequences = train_data['sequence'].tolist()
train_labels = train_data['label'].tolist()

val_sequences = val_data['sequence'].tolist()
val_labels = val_data['label'].tolist()

test_sequences = test_data['sequence'].tolist()
test_labels = test_data['label'].tolist()


# In[8]:


# In [8]:
### TOKENINIZING THE DATASET ###
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("/hpctmp/e0543831/GPU/InstaDeepAI")


# In[9]:


# In [9]:
# Convert the split dataframes to Hugging Face Datasets
train_dataset = Dataset.from_dict({"data": train_sequences, 'labels': train_labels})
val_dataset = Dataset.from_dict({"data": val_sequences, 'labels': val_labels})
test_dataset = Dataset.from_dict({"data": test_sequences, 'labels': test_labels})


# In[10]:


# In [10]:
def tokenize_function(examples):
    outputs = tokenizer(examples["data"])
    return outputs

# Tokenize the datasets
tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["data"],
)

tokenized_val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["data"],
)

tokenized_test_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["data"],
)


# In[11]:


# In [11]:
batch_size = 64
model_name = 'NLP_OLD_NGS_DATA_' + str(num_random_rows) +'_6learningrate_8000steps_2_Epochs_Data_MCC'
args = TrainingArguments(
    f"{model_name}-finetuned-NucleotideTransformer",
    remove_unused_columns=False,
    evaluation_strategy="steps",
    save_strategy="steps",
    learning_rate=1e-6,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="mcc_score",
    label_names=["labels"],
    dataloader_drop_last=True,
    max_steps=8000
)


# In[12]:


# In [12]:
"""Next, we define the metric we will use to evaluate our models and write a `compute_metrics` function. We can load this from the `scikit-learn` library."""
def compute_metrics_mcc(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids
    r = {'mcc_score': matthews_corrcoef(references, predictions)}
    return r


# In[13]:


# In [13]:
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_mcc,
)


# In[14]:


# In [14]:
"""We can now finetune our model by just calling the `train` method:"""
train_results = trainer.train()

print(train_results)


# In[ ]:


# In [15]:
"""As with the first task, the time can be greatly reduced by increasing the batch size.

#### **Validation MCC score**
"""

curve_evaluation_mcc_score = [[a['step'], a['eval_mcc_score']] for a in trainer.state.log_history if 'eval_mcc_score' in a.keys()]
eval_mcc_score = [c[1] for c in curve_evaluation_mcc_score]
steps = [c[0] for c in curve_evaluation_mcc_score]

plt.plot(steps, eval_mcc_score, 'b', label='Validation MCC score')
plt.title('Validation MCC score for enhancer prediction')
plt.xlabel('Number of training steps performed')
plt.ylabel('Validation MCC score')
plt.legend()
plt.show()


# In[ ]:


# In [16]:
# """#### **MCC on the test dataset**"""

# # Compute the MCC score on the test dataset :
print(f"MCC score on the test dataset: {trainer.predict(tokenized_test_dataset).metrics['test_mcc_score']}")


# In[ ]:


# In [17]:
#### **Validation F1 score**
#curve_evaluation_f1_score =[[a['step'],a['eval_f1_score']] for a in trainer.state.log_history if 'eval_f1_score' in a.keys()]
#eval_f1_score = [c[1] for c in curve_evaluation_f1_score]
#steps = [c[0] for c in curve_evaluation_f1_score]

#plt.plot(steps, eval_f1_score, 'b', label='Validation F1 score')
#plt.title('Validation F1 score for promoter prediction')
#plt.xlabel('Number of training steps performed')
#plt.ylabel('Validation F1 score')
#plt.legend()
#plt.show()
# Compute the F1 score on the test dataset :
#print(f"F1 score on the test dataset: {trainer.predict(tokenized_test_dataset).metrics['test_f1_score']}")
     


# In[ ]:


# Define the directory paths to save the model and tokenizer
output_dir = 'trained_model_GPU_' + model_name
tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)


# In[ ]:


# Load the saved tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = AutoModelForSequenceClassification.from_pretrained(output_dir)


# In[ ]:


# Load the data, either FNS from Jingyun or Generated from Sangeetha

FNS_dataframe = pd.read_csv('FNS Sequences for ML model validation.csv')
# FNS_dataframe = pd.read_csv('Sangeetha_Seq_for_verification.csv')

# Data preprocessing: Extracting the DNA samples
FNS_dataframe = FNS_dataframe.dropna()
FNS_samples = FNS_dataframe['Sample']
FNS_samples.reset_index(drop=True, inplace=True)
FNS_sequence = FNS_dataframe['DNA Sequence']
FNS_sequence.reset_index(drop=True, inplace=True)

# Extract sequences from the DataFrame
new_sequences = FNS_sequence.tolist()

# Tokenize the new sequences
tokenized_sequences = tokenizer(new_sequences, return_tensors="pt", padding=True, truncation=True)


# In[ ]:


# Perform inference
model.eval()
with torch.no_grad():
    outputs = model(**tokenized_sequences)

# Get the predicted labels
predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()
class_mapping = {0: 'High', 1: 'Medium', 2: 'Low'}
predicted_labels = [class_mapping[idx] for idx in predicted_labels]
# Add the predicted labels to the DataFrame
FNS_dataframe['Predicted_Labels'] = predicted_labels

# Save the DataFrame with predicted labels
FNS_dataframe.to_csv(output_dir + '_FNS_Predictions.csv', index=False)


# In[ ]:


print(predicted_labels)


# In[ ]:


x = pd.read_csv('JY_Exp_Results.csv')
Count_Correct = 0
for i in range(len(x)):
    if x['Exp Classification'][i] == predicted_labels[i]:
        Count_Correct +=1 

Correct_Percentage = Count_Correct/len(x) * 100
print("Percentage of correct predictions using Jing Yun's Data: " + str(Correct_Percentage))


# In[ ]:


predict = trainer.predict(tokenized_test_dataset)
prediction = []
for i in predict[0]:
    prediction.append(np.array(i).argmax())

prediction = [int(i) for i in prediction]
class_mapping = {0: 'High', 1: 'Medium', 2: 'Low'}
predicted_labels = [class_mapping[idx] for idx in prediction]
predicted_labels_series = pd.Series(predicted_labels, name='Predicted_Labels')


new_data_with_predictions = pd.concat([FNS_samples, predicted_labels_series], axis=1)



# In[ ]:


y_test2 = [int(i) for i in tokenized_test_dataset['labels']]
# Dictionary to store counts of integers
count_dict = {}
count_dict[0] = 0
count_dict[1] = 0
count_dict[2] = 0

# Count occurrences of integers in the list
for i in y_test2:
    if i==0:
        count_dict[0] += 1
    elif i==1:
        count_dict[1] += 1
    elif i==2:
        count_dict[2] += 1

# Print the counts of integers
for num, count in count_dict.items():
    print(f"The integer {num} appears {count} time(s) in the list.")
    
n_High,n_Medium,n_Low = count_dict.values()
count_High,count_Medium,count_Low = 0,0,0

#n_High,n_Low = count_dict.values()
#count_High,count_Low = 0,0

for i in range(len(prediction)):
    if prediction[i] == y_test2[i] and prediction[i] == 0:
        count_High += 1
    elif prediction[i] == y_test2[i] and prediction[i] == 1:
        count_Medium += 1
    elif prediction[i] == y_test2[i] and prediction[i] == 2:
        count_Low += 1

Correct_High = count_High/n_High * 100
Correct_Medium = count_Medium/n_Medium * 100
Correct_Low = count_Low/n_Low * 100

results_RF2 = {'Correctly Predicted-High': Correct_High,
           'Correctly Predicted-Medium': Correct_Medium, \
           'Correctly Predicted-Low':Correct_Low}
print(results_RF2)


# In[ ]:


with open(output_dir + '_accuracy', 'w') as f:
    for key, value in results_RF2.items(): 
        f.write('%s:%s\n' % (key, value))


# In[ ]:




