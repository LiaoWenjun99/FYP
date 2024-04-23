# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:46:45 2023

@author: Wenjun
"""
import tensorflow as tf
from tensorflow import keras
import sys
import pandas as pd
import numpy as np
import time
from sklearn.metrics import classification_report


features = '500'
title = 'DL_AA_adam_3_layers_' + features 


print('Importing necessary python libraries...')
### Importing in data ###

start_import_time = time.time() 
samples_dataframe_H = pd.read_csv('data_AA_H.csv',names = ['0'])
samples_dataframe_M = pd.read_csv('data_AA_M.csv',names = ['0'])
samples_dataframe_L = pd.read_csv('data_AA_L.csv',names = ['0'])
end_import_time = time.time()
import_time = end_import_time-start_import_time
print('Importing time =', import_time)

# define number of samples to extract
num_random_rows = 420000
H_rand = samples_dataframe_H.sample(n=num_random_rows)

M_rand = samples_dataframe_M.sample(n=num_random_rows)

L_rand = samples_dataframe_L.sample(n=num_random_rows)

samples_all = pd.concat([H_rand, M_rand, L_rand])


print('High')
print(len(samples_dataframe_H),'\n')
print('Medium')
print(len(samples_dataframe_M),'\n')
print('Low')
print(len(samples_dataframe_L),'\n')

# In[ ]:

# Selecting column '0' and splitting it into columns 
samples_all = samples_all['0'].str.split('',expand=True)
# Removing the first col of 0 followed by last col
samples_all = samples_all.iloc[:,1:]
samples_all.drop(columns=samples_all.columns[-1],axis=1,inplace=True)

#filtering just mutated regions
samples_all = samples_all.loc[:,83:249]
# In[]:

# One hot encoding on all columns to separate into e.g 3_C,4_T, 4_G, 5_A
start_ohe_time = time.time()
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

##all_ohe = H_ohe
##input_X = all_ohe.loc[:,H_ohe.columns != 'class'] 
##input_X = input_X.sample()
##input_X.to_csv( title + '_cols_OHE.csv', index=False)

# H_ohe['class'] = H_perflist.flatten().tolist()
# eg = H_ohe.iloc[0:2]
### Separation in train and test set  ###


from sklearn.model_selection import train_test_split

start_split_test_train = time.time()
all_ohe = H_ohe
input_X = all_ohe.loc[:,H_ohe.columns != 'class'] 
input_Y = all_ohe['class']
X_train, X_test, Y_train, Y_test = train_test_split(input_X,input_Y,train_size=0.7,random_state=0) 
end_split_test_train = time.time()
t_split_test_train = end_split_test_train - start_split_test_train
print('train test split: ',t_split_test_train)

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


start_RF_fit = time.time()

clf_RFclass = RandomForestClassifier()
clf_RFclass.fit(X_train,Y_train)

end_RF_fit = time.time()
t_RF_fit = end_RF_fit - start_RF_fit
print('Fit time: ',t_RF_fit)


results_RF = {'total no. of samples': len(all_ohe.axes[0]),
           'total no. of features': len(all_ohe.axes[1]), \
           'train/test split':'70/30',\
           'train test split time':t_split_test_train,\
           'fit time' : str(t_RF_fit),\
           
           
          }

with open(title + '_RFC_Results.txt', 'w') as f:
    for key, value in results_RF.items(): 
        f.write('%s:%s\n' % (key, value))
f.close()

# In[22]:

start_Y_prediction_time = time.time()

Y_pred = clf_RFclass.predict(X_test)
report = classification_report(Y_test,Y_pred,output_dict=True)

end_Y_prediction_time = time.time()
Y_prediction_time = end_Y_prediction_time - start_Y_prediction_time
print('Y_prediction_time: ',Y_prediction_time)

#report

report_df = pd.DataFrame(report)
report_df.to_csv('Classification_report_WO_Feat_Sel_' + features + '.csv')


# Feature Importance
feat_imp = clf_RFclass.feature_importances_
feat_imp_df = pd.DataFrame(feat_imp)

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


start_split_test_train2 = time.time()

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(input_X2,input_Y2,test_size=0.3,random_state=0)

end_split_test_train2 = time.time()
t_split_test_train2 = end_split_test_train2 - start_split_test_train2
print('train test split 2: ',t_split_test_train2)

start_RF_fit2 = time.time()


clf_RFclass2 = RandomForestClassifier()

clf_RFclass2.fit(X_train2,Y_train2)


end_RF_fit2 = time.time()
t_RF_fit2 = end_RF_fit2 - start_RF_fit2
print('Feature Importance ', len(extract_top_features), ' Fit time: ',t_RF_fit2)


results_RF2 = {'total no. of samples': len(all_ohe_extracted.axes[0]),
           'total no. of features': len(extract_top_features), \
           'train/test split':'70/30',\
           'train test split time':t_split_test_train2,\
           'fit time' : str(t_RF_fit2),\
           
           
          }

start_Y_prediction_time = time.time()

Y_pred2 = clf_RFclass2.predict(X_test2)
report2 = classification_report(Y_test2,Y_pred2,output_dict=True)

end_Y_prediction_time = time.time()
Y_prediction_time_feature_importance = end_Y_prediction_time - start_Y_prediction_time
print('Feature Importance', len(extract_top_features), 'prediction time: ',Y_prediction_time_feature_importance)
#report
report_df2 = pd.DataFrame(report2)
report_df2.to_csv('Classification_report_RFC_'+ title +'.csv')

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

# Creating a model using Sequential API
# # Classification MLP using 2 hidden layers
# model = keras.models.Sequential()  # Sequential model created
# model.add(keras.layers.Flatten(input_shape=[28, 28])) 
# # adding a Flatten layer: convert each input image into
# # a 1D array (it does not have any parameters, but it the first layer)
# # ALternatively, can add keras.layers.InputLayer and set shape = [28,28]
# model.add(keras.layers.Dense(300, activation="relu")) # Dense hidden layer with 300 neurons, using ReLU activation function
# model.add(keras.layers.Dense(100, activation="relu")) # second dense layer
# model.add(keras.layers.Dense(10, activation="softmax")) # Dense output layer with 10 neurons (ONE PER CLASS) using SOFTMAX activation function

### OR instead of adding one by one can just do this ###
start_fit = time.time()

model = keras.models.Sequential([
 keras.layers.Dense(len(H_ohe.columns), activation="relu"),
 # keras.layers.Dropout(0.5),
 keras.layers.Dense(2/3 * len(H_ohe.columns), activation="relu"),
 # keras.layers.Dropout(0.5),
 keras.layers.Dense(len(class_names), activation="softmax")
])


# All the parameters of a layer can be accessed using its get_weights() and
# set_weights() method. For a Dense layer, this includes both the connection weights and the bias terms: 
# hidden1 = model.get_layer('dense_13')
# weights, biases = hidden1.get_weights()

### After a model is created, you must call its compile() method to specify the loss function and the optimizer to use. ###
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# Sparse labels means that for each instance, there is just a target class index, from 0 to 3 in this case
# "sgd": Stochastic Gradient Descent. 

# loss = categorical_crossentrophy can be used for one-hot vectors such as [0,0,3] to represent class 3
# sigmoid activation for binary classification instead of softmax if wanna use binary_crossentropy loss

# If you want to convert sparse labels (i.e., class indices) to one-hot
# vector labels, you can use the keras.utils.to_categorical()
# function. To go the other way round, you can just use the np.arg
# max() function with axis=1.

### Training and Evaluating the Model ###

history = model.fit(X_train, y_train, epochs=30,validation_data=(X_valid, y_valid), verbose =0 )
# Keras will measure the loss and the extra metrics on this set at the
# end of each epoch, which is very useful to see how well the model really performs: if
# the performance on the training set is much better than on the validation set, your
# model is probably overfitting the training set (or there is a bug, such as a data mis‐
# match between the training set and the validation set)
end_fit = time.time()
t_fit = end_fit - start_fit
print('DL Fitting Time: ',t_fit)

### PLOT LEARNING CURVES ###
import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.savefig(title + '_Accuracy & Loss Plot.jpg')
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
           
with open('DL Results_' + title, 'w') as f:
    for key, value in results_RF2.items(): 
        f.write('%s:%s\n' % (key, value))
f.close()

### Saving the Model ###
full_file_name = title + '_model' + '.h5'
model.save(full_file_name)




