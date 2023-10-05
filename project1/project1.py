import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data=pd.read_csv('ACME-HappinessSurvey2020.csv')
#train = data[data.train==True]
X = data.drop(columns=['Y'])
Y = data.Y

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=1)

print(X.shape, Y.shape)

from sklearn.neighbors import KNeighborsClassifier
import time, math
from sklearn.metrics import accuracy_score
cols_results=['family','model','classification_rate','runtime']
results = pd.DataFrame(columns=cols_results)
kVals = range(1,10)
knn_names = ['KNN-'+str(k) for k in kVals]
for k in kVals:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    time_start = time.time()
    y_pred = knn.predict(X_test)
    time_run = time.time()-time_start
    
    results = results.append(pd.DataFrame([['KNN',knn_names[k-1],accuracy_score(y_test,y_pred),time_run]],columns=cols_results),ignore_index=True)
results[results.family=='KNN']
