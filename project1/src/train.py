import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import KFold, cross_val_score
from imblearn.under_sampling import RandomUnderSampler  


data=pd.read_csv("C:\\Users\\agnes\\Documents\\apziva_ai_residency\\project2\\data\\term-deposit-marketing-2020.csv")


# Data encoding
data['job']=data['job'].astype('category')
data['marital']=data['marital'].astype('category')
data['education']=data['education'].astype('category')
data['default']=data['default'].astype('category')
data['housing']=data['housing'].astype('category')
data['loan']=data['loan'].astype('category')
data['contact']=data['contact'].astype('category')
data['month']=data['month'].astype('category')
data['y']=data['y'].astype('category')
cat_columns = data.select_dtypes(['category']).columns
data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
 # not using loan or default 
x  = data[['age', 'job','marital','education','balance','housing','day','month','duration','campaign']]
y = data[['y']]

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=0)

# under sampling to account for data imbalance 
under_sampler = RandomUnderSampler()
X_res, y_res = under_sampler.fit_resample(x, y)
x_train_res, y_train_res=under_sampler.fit_resample(x_train,y_train)

# time for grid search and logistic regression 

f1 = make_scorer(f1_score , average='macro')
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10,scoring=f1)
logreg_cv.fit(x_train_res, y_train_res.values.ravel())


# Evaluation
logreg2=LogisticRegression(C=0.01,penalty="l2")
logreg2.fit(x_train_res, y_train_res.values.ravel())

predictions=lr2.predict(x_test)
f1_score(y_test,predictions)

with open('trained_model.pkl', 'wb') as f:
    pickle.dump(dt, f)