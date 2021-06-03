import pandas as pd 
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

data = pd.read_csv("./mushroom.csv")
label =LabelEncoder()
for i in data.columns:
    data[i] = label.fit_transform(data[i])

X = data.drop(columns=['type'])
Y = data.type

X1=pd.get_dummies(X)
print

fs = SelectKBest(chi2, k='all')
X_new = fs.fit_transform(X, Y)
print(X_new)
# plot the scores
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
# X = df.drop(columns=['cap-shape','cap-color','gill-attachment','stalk-shape','veil-type', 'veil-color', 'ring-number'])



