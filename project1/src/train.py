

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix, classification_report
import pickle


data=pd.read_csv("C:\\Users\\agnes\\Documents\\apziva_ai_residency\\project1\\Data\\ACME-HappinessSurvey2020.csv")

# Prepare data
y = data["Y"]
X = data.drop(columns = ['Y'])
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=12)


dt = DecisionTreeClassifier(random_state=0, ccp_alpha=0.016)
dt.fit(X_train,Y_train)

# Evaluation
pred_test = dt.predict(X_test)
accuracy_score(Y_test, pred_test)



with open('trained_model.pkl', 'wb') as f:
    pickle.dump(dt, f)