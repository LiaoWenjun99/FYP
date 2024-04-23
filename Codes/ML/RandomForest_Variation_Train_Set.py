#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries

import sys
import pandas as pd
import numpy as np

import time

from sklearn.metrics import classification_report

print('Importing necessary python libraries...')

title = 'RFC_Varying_Train_Set'
# # Sample  

# In[ ]:

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

# define number of samples to extract
num_random_rows = 420000
#H_rand = samples_dataframe_H[0:num_random_rows]
#M_rand = samples_dataframe_M[0:num_random_rows]
#L_rand = samples_dataframe_L[0:num_random_rows]


H_rand = samples_dataframe_H.sample(n=num_random_rows)
H_rand = H_rand[~H_rand['0'].str.contains('N')]

M_rand = samples_dataframe_M.sample(n=num_random_rows)
M_rand = M_rand[~M_rand['0'].str.contains('N')]

L_rand = samples_dataframe_L.sample(n=num_random_rows)
L_rand = L_rand[~L_rand['0'].str.contains('N')]
samples_all = pd.concat([H_rand, M_rand, L_rand])
#samples_all = pd.concat([M_rand, L_rand])
#samples_all = pd.concat([M_rand, L_rand])
print('High')
print(len(H_rand),'\n')
print('Medium')
print(len(M_rand),'\n')
print('Low')
print(len(L_rand),'\n')



# In[ ]:

# Selecting column '0' and splitting it into columns 
samples_all = samples_all['0'].str.split('',expand=True)
# Removing the first col of 0 followed by last col
samples_all = samples_all.iloc[:,1:]
samples_all.drop(columns=samples_all.columns[-1],axis=1,inplace=True)

#samples_all.to_csv('samples_RandomForest_all.csv')


# In[11]:


#samples_all = samples_all.loc[:,83:249]


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

# H_ohe['class'] = H_perflist.flatten().tolist()
##all_ohe = H_ohe
##sample = all_ohe.loc[:,H_ohe.columns != 'class'] 
##sample = sample.sample()
##sample.to_csv(title + '_cols_OHE.csv', index=False)

# In[14]:


#H_ohe.to_csv('one hot encoding_RandomForest_all.csv')


# In[6]:


from sklearn.model_selection import train_test_split

start_split_test_train = time.time()
all_ohe = H_ohe
input_X = all_ohe.loc[:,H_ohe.columns != 'class'] 
input_Y = all_ohe['class']
X_train, X_test, Y_train, Y_test = train_test_split(input_X,input_Y,train_size=0.35,random_state=0)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(input_X,input_Y,train_size=0.28,random_state=0)
X_train3, X_test3, Y_train3, Y_test3 = train_test_split(input_X,input_Y,train_size=0.21,random_state=0)
X_train4, X_test4, Y_train4, Y_test4 = train_test_split(input_X,input_Y,train_size=0.14,random_state=0)
X_train5, X_test5, Y_train5, Y_test5 = train_test_split(input_X,input_Y,train_size=0.07,random_state=0)
end_split_test_train = time.time()
t_split_test_train = end_split_test_train - start_split_test_train
print('train test split: ',t_split_test_train)


# In[7]:


#X_train.to_csv('X_train_RandomForest_all.csv')
#X_test.to_csv('X_test_RandomForest_all.csv')
#Y_train.to_csv('Y_train_RandomForest_all.csv')
#Y_test.to_csv('Y_test_RandomForest_all.csv')


# In[8]:


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


# In[10]:


start_RF_fit = time.time()

clf_RFclass = RandomForestClassifier()
clf_RFclass.fit(X_train,Y_train)
clf_RFclass2 = RandomForestClassifier()
clf_RFclass2.fit(X_train2,Y_train2)
clf_RFclass3 = RandomForestClassifier()
clf_RFclass3.fit(X_train3,Y_train3)
clf_RFclass4 = RandomForestClassifier()
clf_RFclass4.fit(X_train4,Y_train4)
clf_RFclass5 = RandomForestClassifier()
clf_RFclass5.fit(X_train5,Y_train5)

end_RF_fit = time.time()
t_RF_fit = end_RF_fit - start_RF_fit
print('Fit time: ',t_RF_fit)


results_RF = {'total no. of samples': len(all_ohe.axes[0]),
           'total no. of features': len(all_ohe.axes[1]), \
           'train/test split':'70/30',\
           'train test split time':t_split_test_train,\
           'fit time' : str(t_RF_fit),\
           
           
          }

with open('RandomForest_all.txt', 'w') as f:
    for key, value in results_RF.items(): 
        f.write('%s:%s\n' % (key, value))
f.close()

# In[22]:

start_Y_prediction_time = time.time()

Y_pred = clf_RFclass.predict(X_test)
report = classification_report(Y_test,Y_pred,output_dict=True)
Y_pred2 = clf_RFclass2.predict(X_test2.iloc[0:126000, :])
report2 = classification_report(Y_test2.iloc[0:126000],Y_pred2,output_dict=True)
Y_pred3 = clf_RFclass3.predict(X_test3.iloc[0:126000, :])
report3 = classification_report(Y_test3.iloc[0:126000],Y_pred3,output_dict=True)
Y_pred4 = clf_RFclass4.predict(X_test4.iloc[0:126000, :])
report4 = classification_report(Y_test4.iloc[0:126000],Y_pred4,output_dict=True)
Y_pred5 = clf_RFclass5.predict(X_test5.iloc[0:126000, :])
report5 = classification_report(Y_test5.iloc[0:126000],Y_pred5,output_dict=True)

end_Y_prediction_time = time.time()
Y_prediction_time = end_Y_prediction_time - start_Y_prediction_time
print('Y_prediction_time: ',Y_prediction_time)

#report

report_df = pd.DataFrame(report)
report_df.to_csv('Classification report_RandomForest6_same_test_number.csv')
report_df2 = pd.DataFrame(report2)
report_df2.to_csv('Classification report_RandomForest7_same_test_number.csv')
report_df3 = pd.DataFrame(report3)
report_df3.to_csv('Classification report_RandomForest8_same_test_number.csv')
report_df4 = pd.DataFrame(report4)
report_df4.to_csv('Classification report_RandomForest9_same_test_number.csv')
report_df5 = pd.DataFrame(report5)
report_df5.to_csv('Classification report_RandomForest10_same_test_number.csv')

# In[34]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

##start_others_RF_time = time.time()
##
##confusionmatrix_RF = confusion_matrix(Y_test, Y_pred)
##disp = ConfusionMatrixDisplay(confusion_matrix = confusionmatrix_RF)
##disp = disp.plot()
###plt.show()
##plt.savefig('Confusion Matrix_RandomForest.jpg')
##

# In[1]:


from sklearn.metrics import accuracy_score
##
##score_clf_test = clf_RFclass.score(X_test,Y_test)
##score_clf_train = clf_RFclass.score(X_train,Y_train)
##
##metric_train = accuracy_score(y_true = Y_train, y_pred = clf_RFclass.predict(X_train) )
##metric_test = accuracy_score(y_true = Y_test, y_pred = Y_pred )



# In[ ]:

##
##cv_scores_logis = cross_val_score(clf_RFclass,X_train,Y_train,cv=5)
##cv_scores_mean_logis = cv_scores_logis.mean()
##cv_scores_sd_logis = cv_scores_logis.std()


# In[ ]:

##
##others_RF = {
##           'clf score test':score_clf_test,\
##    'clf score train': score_clf_train,\
##    'metric score test' : metric_test,\
##    'metric score train': metric_train,\
##    'CV folds':'5',\
##                     'CV scores mean': cv_scores_mean_logis,\
##                     'CV scores sd': cv_scores_sd_logis,\
##                     'CV scores':cv_scores_logis\
##           
##           
##          }
##
##with open('RandomForest_all_others.txt', 'w') as k:
##    for key, value in others_RF.items(): 
##        k.write('%s:%s\n' % (key, value))
##k.close()
##
##end_others_RF_time = time.time()
##others_RF_time = end_others_RF_time-start_others_RF_time
##print('Others_RF_time: ',others_RF_time)
##

### In[ ]:
##
##start_feat_impt_time = time.time()
##
##feat_imp = clf_RFclass.feature_importances_
##feat_imp_df = pd.DataFrame(feat_imp)
##feat_imp_df.to_csv('feat_imp_RandomForest.csv')
##
##s = sorted(enumerate(feat_imp), key=lambda x: x[1], reverse= True)
##extract_top_features = []
##for i in range(0,652):
##    extract_top_features.append(s[i][0])
##extract_top_features.append(len(H_ohe.columns)-1)
##all_ohe_extract = H_ohe
##all_ohe_extracted = all_ohe_extract.iloc[:, extract_top_features]
##
##input_X2 = all_ohe_extracted.loc[:,all_ohe_extracted.columns != 'class'] 
##input_Y2 = all_ohe_extracted['class']
##
##end_feat_impt_time = time.time()
##feat_impt_time = end_feat_impt_time - start_feat_impt_time
##print('Feature importance sorting time: ',feat_impt_time)
##
##
##start_split_test_train2 = time.time()
##
##X_train2, X_test2, Y_train2, Y_test2 = train_test_split(input_X2,input_Y2,test_size=0.3,random_state=0)
##
##end_split_test_train2 = time.time()
##t_split_test_train2 = end_split_test_train2 - start_split_test_train2
##print('train test split 2: ',t_split_test_train2)
##
##start_RF_fit2 = time.time()
##
##
##clf_RFclass2 = RandomForestClassifier()
##
##clf_RFclass2.fit(X_train2,Y_train2)
##
##
##end_RF_fit2 = time.time()
##t_RF_fit2 = end_RF_fit2 - start_RF_fit2
##print('Feature Importance ', len(extract_top_features), ' Fit time: ',t_RF_fit2)
##
##
##results_RF2 = {'total no. of samples': len(all_ohe_extracted.axes[0]),
##           'total no. of features': len(extract_top_features), \
##           'train/test split':'70/30',\
##           'train test split time':t_split_test_train2,\
##           'fit time' : str(t_RF_fit2),\
##           
##           
##          }
##
##start_Y_prediction_time = time.time()
##
##Y_pred2 = clf_RFclass2.predict(X_test2)
##report2 = classification_report(Y_test2,Y_pred2,output_dict=True)
##
##end_Y_prediction_time = time.time()
##Y_prediction_time_feature_importance = end_Y_prediction_time - start_Y_prediction_time
##print('Feature Importance', len(extract_top_features), 'prediction time: ',Y_prediction_time_feature_importance)
###report
##report_df2 = pd.DataFrame(report2)
##report_df2.to_csv('Classification report_RandomForest_After_Feature_Selection_top_1000_features.csv')

### In[ ]:
##    
##extract_top_features = []
##for i in range(0,1999):
##    extract_top_features.append(s[i][0])
##extract_top_features.append(len(H_ohe.columns)-1)
##all_ohe_extract = H_ohe
##all_ohe_extracted = all_ohe_extract.iloc[:, extract_top_features]
##
##input_X3 = all_ohe_extracted.loc[:,all_ohe_extracted.columns != 'class'] 
##input_Y3 = all_ohe_extracted['class']
##
##start_split_test_train3 = time.time()
##
##X_train3, X_test3, Y_train3, Y_test3 = train_test_split(input_X3,input_Y3,test_size=0.3,random_state=0) 
##end_split_test_train3 = time.time()
##t_split_test_train3 = end_split_test_train3 - start_split_test_train3
##print('train test split: ',t_split_test_train3)
##
##start_RF_fit3 = time.time()
##
##
##clf_RFclass3 = RandomForestClassifier()
##
##clf_RFclass3.fit(X_train3,Y_train3)
##
##
##end_RF_fit3 = time.time()
##t_RF_fit3 = end_RF_fit3 - start_RF_fit3
##
##
##results_RF3 = {'total no. of samples': len(all_ohe_extracted.axes[0]),
##           'total no. of features': len(extract_top_features), \
##           'train/test split':'70/30',\
##           'train test split time':t_split_test_train3,\
##           'fit time' : str(t_RF_fit3),\
##           
##           
##          }
##    
##Y_pred3 = clf_RFclass3.predict(X_test3)
##report3 = classification_report(Y_test3,Y_pred3,output_dict=True)
###report
##report_df3 = pd.DataFrame(report3)
##report_df3.to_csv('Classification report_RandomForest_After_Feature_Selection_top_500_features.csv')
# In[ ]:
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
##confusionmatrix_RF2 = confusion_matrix(Y_test2, Y_pred2)
##disp2 = ConfusionMatrixDisplay(confusion_matrix = confusionmatrix_RF2)
##disp2 = disp2.plot()
###plt.show()
##plt.savefig('Confusion Matrix_RandomForest_After_Feature_Selection_top_1000_features.jpg')

##confusionmatrix_RF3 = confusion_matrix(Y_test3, Y_pred3)
##disp3 = ConfusionMatrixDisplay(confusion_matrix = confusionmatrix_RF3)
##disp3 = disp3.plot()
###plt.show()
##plt.savefig('Confusion Matrix_RandomForest_After_Feature_Selection_top_500_features.jpg')

from sklearn.metrics import accuracy_score

##score_clf_test2 = clf_RFclass2.score(X_test2,Y_test2)
##score_clf_train2 = clf_RFclass2.score(X_train2,Y_train2)
##
##metric_train2 = accuracy_score(y_true = Y_train2, y_pred = clf_RFclass2.predict(X_train2) )
##metric_test2 = accuracy_score(y_true = Y_test2, y_pred = Y_pred2 )
#### REPEAT WITH DIFF NUMBER OF FEATURES #####
##score_clf_test3 = clf_RFclass3.score(X_test3,Y_test3)
##score_clf_train3 = clf_RFclass3.score(X_train3,Y_train3)
##
##metric_train3 = accuracy_score(y_true = Y_train3, y_pred = clf_RFclass3.predict(X_train3) )
##metric_test3 = accuracy_score(y_true = Y_test3, y_pred = Y_pred3 )


# In[ ]:


##cv_scores_logis2 = cross_val_score(clf_RFclass2,X_train2,Y_train2,cv=5)
##cv_scores_mean_logis2 = cv_scores_logis2.mean()
##cv_scores_sd_logis2 = cv_scores_logis2.std()

#### REPEAT WITH DIFF NUMBER OF FEATURES #####

##cv_scores_logis3 = cross_val_score(clf_RFclass3,X_train3,Y_train3,cv=5)
##cv_scores_mean_logis3 = cv_scores_logis3.mean()
##cv_scores_sd_logis3 = cv_scores_logis3.std()

# In[ ]:


##others_RF2 = {
##           'clf score test':score_clf_test2,\
##    'clf score train': score_clf_train2,\
##    'metric score test' : metric_test2,\
##    'metric score train': metric_train2,\
##    'CV folds':'5',\
##                     'CV scores mean': cv_scores_mean_logis2,\
##                     'CV scores sd': cv_scores_sd_logis2,\
##                     'CV scores':cv_scores_logis2\
##           
##           
##          }
##
##with open('RandomForest_all_others2.txt', 'w') as k:
##    for key, value in others_RF2.items(): 
##        k.write('%s:%s\n' % (key, value))
##k.close()

##others_RF3 = {
##           'clf score test':score_clf_test3,\
##    'clf score train': score_clf_train3,\
##    'metric score test' : metric_test3,\
##    'metric score train': metric_train3,\
##    'CV folds':'5',\
##                     'CV scores mean': cv_scores_mean_logis3,\
##                     'CV scores sd': cv_scores_sd_logis3,\
##                     'CV scores':cv_scores_logis3\
##           
##           
##          }
##
##with open('RandomForest_all_others2.txt', 'w') as k:
##    for key, value in others_RF3.items(): 
##        k.write('%s:%s\n' % (key, value))
##
##k.close()


####save the trained model

##import joblib
##
##start_saving_model_time = time.time()
##joblib.dump(clf_RFclass, "./RFC_ALL.joblib")
##end_saving_model_time = time.time()
##saving_model_time_wo_feature_impt = end_saving_model_time - start_saving_model_time
##print("Saving model time without feat impt: ", saving_model_time_wo_feature_impt)
##
##
##start_saving_model_time2 = time.time()
##joblib.dump(clf_RFclass2, "./RFC_Feature_Selection_Top_1500.joblib")
##end_saving_model_time2 = time.time()
##saving_model_time_wth_1000_feature_impt = end_saving_model_time2 - start_saving_model_time2
##print("Saving model time without feat impt: ", saving_model_time_wth_1000_feature_impt)
##joblib.dump(clf_RFclass3, "./RFC_Feature_Selection_Top_2000.joblib")

##import pickle
##
### save the iris classification model as a pickle file
##model_pkl_file = title + '.pkl'
##
##pickle.dump(clf_RFclass, open('RFC_model.pkl', 'wb'))



