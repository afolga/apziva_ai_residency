

import pickle
import pandas as pd
with open('trained_model.pkl', 'rb') as f:
    dt = pickle.load(f)


def predict(model, new_data):
    predictions = model.predict(new_data)
    return predictions



new_data=pd.read_csv("C:\\Users\\agnes\\Documents\\apziva_ai_residency\\project1\\Data\\ACME-HappinessSurvey2020.csv")
y = new_data['Y']
new_data.drop('Y', axis=1, inplace=True)



predictions = predict(dt, new_data)
predictions

# array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
#        0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0,
#        1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0,
#        0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0,
#        0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1,
#        1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1], dtype=int64)