# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:46:45 2023

@author: Wenjun
"""
### Import relevant libraries
import tensorflow as tf
from tensorflow import keras
import sys
import pandas as pd
import numpy as np
import time

### State number of features for feature selection and title
features = '500'
title = 'adam_3_layers_Feat_Sel_' + features

### Importing in data ###
print('Importing necessary python libraries...')

start_import_time = time.time() 
samples_dataframe_H = pd.read_csv('data_H.csv',names = ['0'])
samples_dataframe_M = pd.read_csv('data_M.csv',names = ['0'])
samples_dataframe_L = pd.read_csv('data_L.csv',names = ['0'])
end_import_time = time.time()
import_time = end_import_time-start_import_time
print('Importing time =', import_time)

#samples_dataframe_H = pd.read_csv('data_AA_H.csv',names = ['0'])
#samples_dataframe_M = pd.read_csv('data_AA_M.csv',names = ['0'])
#samples_dataframe_L = pd.read_csv('data_AA_L.csv',names = ['0'])

print('High')
print(len(samples_dataframe_H),'\n')
print('Medium')
print(len(samples_dataframe_M),'\n')
print('Low')
print(len(samples_dataframe_L),'\n')

# define number of samples to extract
num_random_rows = 420000
#H_rand = samples_dataframe_H[0:num_random_rows]
#M_rand = samples_dataframe_M[0:num_random_rows]
#L_rand = samples_dataframe_L[0:num_random_rows]

### Extract random rows and remove those with N
H_rand = samples_dataframe_H.sample(n=num_random_rows)
H_rand = H_rand[~H_rand['0'].str.contains('N')]

M_rand = samples_dataframe_M.sample(n=num_random_rows)
M_rand = M_rand[~M_rand['0'].str.contains('N')]

L_rand = samples_dataframe_L.sample(n=num_random_rows)
L_rand = L_rand[~L_rand['0'].str.contains('N')]
samples_all = pd.concat([H_rand, M_rand, L_rand])
#samples_all = pd.concat([M_rand, L_rand])
#samples_all = pd.concat([M_rand, L_rand])


# In[ ]:

# Selecting column '0' and splitting it into columns 
samples_all = samples_all['0'].str.split('',expand=True)
# Removing the first col of 0 followed by last col
samples_all = samples_all.iloc[:,1:]
samples_all.drop(columns=samples_all.columns[-1],axis=1,inplace=True)

# In[5]:

start_ohe_time = time.time()

# One hot encoding on all columns to separate into e.g 3_C,4_T, 4_G, 5_A
H_ohe = pd.get_dummies(samples_all)
# Creating labels for H,M,L
H_perflist_H = [0]*len(H_rand)
H_perflist_M = [1]*len(M_rand)
H_perflist_L = [2]*len(L_rand)
H_perflist = H_perflist_H + H_perflist_M + H_perflist_L
#H_perflist = H_perflist_H + H_perflist_L
#H_perflist = H_perflist_M + H_perflist_L 

H_perflist = np.array(H_perflist)
df_temp = pd.DataFrame(H_perflist, columns=['class'], dtype = 'uint8', index = H_ohe.index)
H_ohe = pd.concat([H_ohe,df_temp],axis = 1)

end_ohe_time = time.time()
ohe_time = end_ohe_time-start_ohe_time
print('OHE time =', ohe_time)

#Import feature importance from Random Forest Machine Learning
start_import_time = time.time() 
dataframe_features = pd.read_csv('feat_imp_RandomForest_1000.csv',names = ['0'], skiprows=1)
feat_imp = dataframe_features.to_numpy()

# Perform Feature Selection of n features indicated at the top

s = sorted(enumerate(feat_imp), key=lambda x: x[1], reverse= True)
extract_top_features = []
for i in range(0,int(features)):
    extract_top_features.append(s[i][0])
extract_top_features.append(len(H_ohe.columns)-1)
all_ohe_extract = H_ohe
all_ohe_extracted = all_ohe_extract.iloc[:, extract_top_features]

input_X2 = all_ohe_extracted.loc[:,all_ohe_extracted.columns != 'class'] 
input_Y2 = all_ohe_extracted['class']

# Extract out a sample row of data for prediction
all_ohe2 = all_ohe_extracted
sample2 = all_ohe2.loc[:,all_ohe_extracted.columns != 'class'] 
sample2 = sample2.sample()
sample2.to_csv(title + '_cols_OHE.csv', index=False)

end_import_time = time.time()
import_time = end_import_time-start_import_time
print('Importing time =', import_time)

from sklearn.model_selection import train_test_split
start_split_test_train = time.time()
all_ohe = H_ohe
input_X = input_X2
input_Y = input_Y2
X_train_full, X_test, y_train_full, y_test = train_test_split(input_X,input_Y,train_size=0.7,random_state=0) 
end_split_test_train = time.time()
t_split_test_train = end_split_test_train - start_split_test_train
print('train test split: ',t_split_test_train)

#### Validation Set ###
validation_number = int(1/5 * len(X_train_full))
X_valid, X_train = X_train_full[:validation_number], X_train_full[validation_number:] 
y_valid, y_train = y_train_full[:validation_number], y_train_full[validation_number:]


X_valid, X_train = X_valid.to_numpy(), X_train.to_numpy()
y_valid, y_train = y_valid.to_numpy(), y_train.to_numpy()

#input class names
class_names = ["High","Medium","Low"]
# print("Class names example: ", class_names[y_train[0]])

### Deep Learning Feedforward Neural Network ###

start_fit = time.time()

### Over here, we can increase the nerual layers
### for example to create a 3 layer model, the denomiter must be scaled accordingly

model = keras.models.Sequential([
 keras.layers.Dense(len(H_ohe.columns), activation="relu"),
 # keras.layers.Dropout(0.5),
 keras.layers.Dense(2/3 * len(H_ohe.columns), activation="relu"),
 keras.layers.Dense(1/3 * len(H_ohe.columns), activation="relu"),

 # keras.layers.Dropout(0.5),
 keras.layers.Dense(len(class_names), activation="softmax")
])


### After a model is created, you must call its compile() method to specify the loss function and the optimizer to use. ###
### Over here, the optimizer can be CHANGED to "sgd" for stochastic gradient descent.
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam" , metrics=["accuracy"])

### Training and Evaluating the Model ###

history = model.fit(X_train, y_train, epochs=30,validation_data=(X_valid, y_valid), verbose =0 )

end_fit = time.time()
t_fit = end_fit - start_fit
print('DL Fitting Time: ',t_fit)

### PLOT LEARNING CURVES ###
import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.savefig(title + '.jpg')
plt.show()


### Predictions ###
# X_new = X_test[:1000]
predict = model.predict(X_test)
prediction = []
for i in predict:
    prediction.append(np.array(i).argmax())

prediction = [int(i) for i in prediction]
y_test2 = [int(i) for i in y_test]

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
           'Correctly Predicted-Low':Correct_Low,\
           'DL fit time' : str(t_fit)}
           
with open(title + '.txt', 'w') as f:
    for key, value in results_RF2.items(): 
        f.write('%s:%s\n' % (key, value))
f.close()

### Saving the Model ###
full_file_name = title + '.h5'
model.save(full_file_name)







