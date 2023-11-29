import pickle
import pandas as pd
with open('trained_model.pkl', 'rb') as f:
    dt = pickle.load(f)


def predict(model, new_data):
    predictions = model.predict(new_data)
    return predictions


new_data=pd.read_csv("C:\\Users\\agnes\\Documents\\apziva_ai_residency\\project2\\data\\term-deposit-marketing-2020.csv")
new_data['job']=new_data['job'].astype('category')
new_data['marital']=new_data['marital'].astype('category')
new_data['education']=new_data['education'].astype('category')
new_data['default']=new_data['default'].astype('category')
new_data['housing']=new_data['housing'].astype('category')
new_data['loan']=new_data['loan'].astype('category')
new_data['contact']=new_data['contact'].astype('category')
new_data['month']=new_data['month'].astype('category')
new_data['y']=new_data['y'].astype('category')
cat_columns = new_data.select_dtypes(['category']).columns
new_data[cat_columns] = new_data[cat_columns].apply(lambda x: x.cat.codes)
y = new_data['y']
x  = new_data[['age', 'job','marital','education','balance','housing','day','month','duration','campaign']]




predictions = predict(dt, new_data)
predictions