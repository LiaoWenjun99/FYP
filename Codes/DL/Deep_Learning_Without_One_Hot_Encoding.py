# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 22:01:15 2023

@author: Wenjun
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, optimizers
import time

print('Importing necessary python libraries...')
### Importing in data ###

start_import_time = time.time() 
samples_dataframe_H = pd.read_csv('data_H.csv',names = ['0'])
samples_dataframe_M = pd.read_csv('data_M.csv',names = ['0'])
samples_dataframe_L = pd.read_csv('data_L.csv',names = ['0'])
end_import_time = time.time()
import_time = end_import_time-start_import_time
print('Importing time =', import_time)


print('High')
print(len(samples_dataframe_H),'\n')
print('Medium')
print(len(samples_dataframe_M),'\n')
print('Low')
print(len(samples_dataframe_L),'\n')

# define number of samples to extract
num_random_rows = 400000
H_rand = samples_dataframe_H[0:num_random_rows]
M_rand = samples_dataframe_M[0:num_random_rows]
L_rand = samples_dataframe_L[0:num_random_rows]

samples_all = pd.concat([H_rand, M_rand, L_rand])

# In[ ]:

# Selecting column '0' and splitting it into columns 
# samples_all = samples_all['0'].str.split('',expand=True)
# # Removing the first col of 0 followed by last col
# samples_all = samples_all.iloc[:,1:]
# samples_all.drop(columns=samples_all.columns[-1],axis=1,inplace=True)

# Creating labels for H,M,L
perflist_H = [0]*len(H_rand)
perflist_M = [1]*len(M_rand)
perflist_L = [2]*len(L_rand)
perflist = perflist_H + perflist_M + perflist_L
#H_perflist = H_perflist_H + H_perflist_L
#H_perflist = H_perflist_M + H_perflist_L 

perflist = np.array(perflist)
df_temp = pd.DataFrame(perflist, columns=['class'], dtype = 'uint8', index = samples_all.index)
samples_all = pd.concat([samples_all,df_temp],axis = 1)

# Preprocess the data
encoder = LabelEncoder()
samples_all['efficiency_class_encoded'] = encoder.fit_transform(samples_all['class'])
sequences = samples_all['0'].values
labels = samples_all['efficiency_class_encoded'].values


# Split the data into training and testing sets
sequences_train, sequences_test, labels_train, labels_test = train_test_split(
    sequences, labels, test_size=0.3, random_state=42
)

# Tokenize the DNA sequences
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(sequences_train)

sequences_train = tokenizer.texts_to_sequences(sequences_train)
sequences_test = tokenizer.texts_to_sequences(sequences_test)

# Pad sequences to a fixed length
max_length = max(map(len, sequences_train))
sequences_train = tf.keras.preprocessing.sequence.pad_sequences(sequences_train, maxlen=max_length)
sequences_test = tf.keras.preprocessing.sequence.pad_sequences(sequences_test, maxlen=max_length)

# Build the deep learning model
model = models.Sequential([
    layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_length),
    layers.Conv1D(128, 5, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes: high, medium, low
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(sequences_train, labels_train, epochs=30, validation_split=0.2, verbose =0)

# Evaluate the model
accuracy = model.evaluate(sequences_test, labels_test)[1]
print(f'Test Accuracy: {accuracy}')


# X_new = X_test[:1000]
predict = model.predict(sequences_test)
prediction = []
for i in predict:
    prediction.append(np.array(i).argmax())

prediction = [int(i) for i in prediction]
y_test2 = [int(i) for i in labels_test]

# Dictionary to store counts of integers
count_dict = {}

# Count occurrences of integers in the list
for num in y_test2:
    if num in count_dict:
        count_dict[num] += 1
    else:
        count_dict[num] = 1

# Print the counts of integers
for num, count in count_dict.items():
    print(f"The integer {num} appears {count} time(s) in the list.")
    
n_High,n_Medium,n_Low = count_dict.values()

count_High,count_Medium,count_Low = 0,0,0

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
           
with open('Deep Learning Results', 'w') as f:
    for key, value in results_RF2.items(): 
        f.write('%s:%s\n' % (key, value))
f.close()

### PLOT LEARNING CURVES ###
import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.savefig('Accuracy & Loss Plot.jpg')
plt.show()

