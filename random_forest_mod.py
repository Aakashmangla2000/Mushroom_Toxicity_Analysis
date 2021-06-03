import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data():
        data = pd.read_csv("./mushroom.csv")
        
        label =LabelEncoder()
        for i in data.columns:
            data[i] = label.fit_transform(data[i])
        return data
df = load_data()

def split(df):
    X = df.drop(columns=['type','cap-shape','cap-color','gill-attachment','stalk-shape','veil-type', 'veil-color', 'ring-number'])
    Y = df.type
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
    return X_train,X_test,Y_train,Y_test

X_train,X_test,Y_train,Y_test = split(df)

model = RandomForestClassifier(n_estimators=100,max_depth=5,bootstrap=True)
model.fit(X_train,Y_train)
filename = 'ran_fo_model.sav'
pickle.dump(model, open(filename, 'wb'))