# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 17:07:58 2020

@author: Nagesh
"""

#%%
# importing libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

#%%
# reading the csv file
df = pd.read_csv("spam.csv", encoding = "latin-1")

#%%

df = df.drop(columns = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())

#%%
# converting string values to numeric
df["v1"] = df["v1"].astype('category').cat.codes

#%%

x, y = df["v2"], df["v1"]

#%%

print(y.value_counts())



#%%

# 10 fold cross validation 
confusion_matrices = []
vec = CountVectorizer()

model = MultinomialNB()

cv = KFold(n_splits = 10)
for train_index, test_index in cv.split(x):
    # print(train_index, test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(vec.fit_transform(x_train), y_train)
    confusion_matrices.append(confusion_matrix(y_test, model.predict(vec.transform(x_test))))


#%%

with open("vec_model.pkl", "wb") as f_:
    pickle.dump(vec, f_)

#%%

accuracy_scores = []
for confusion_matrix_ in confusion_matrices:
    accuracy_scores.append(np.trace(confusion_matrix_, dtype = np.int64)*100/(np.sum(confusion_matrix_, dtype = np.int64)))

#%%

print(accuracy_scores)


#%%

accuracy_scores = np.array(accuracy_scores, dtype = "float")
print(np.mean(accuracy_scores))


#%%
# saving the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)


#%%

# testing the model on entire datatset
y_pred  = model.predict(vec.transform(x))
print(confusion_matrix(y, y_pred))

#%%

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
    
with open("vec_model.pkl", "rb") as f_:
    vecc = pickle.load(f_)
    

#%%
model.predict(vecc.transform(["hello world"]))


