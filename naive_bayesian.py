import pandas as pd
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib as plt

data = pd.read_json("1500.json")
ma = np.array(data.as_matrix())
# y_target = ma[:,0]
# X = ma[:,[1,10]]

X = data[['facet_1', 'facet_2', 'facet_3', 'facet_4', 'facet_5', 'facet_6', 'facet_7', 'facet_8', 'facet_9']]
y_target = data['color']

print("# of unique rows: %d" % X.drop_duplicates().shape[0])

gnb = GaussianNB()
y_pred = gnb.fit(X, y_target).predict(X)
y_prob = gnb.predict_proba(X)

print("Percentage of mislabeled points out of a total %d points : %d" % ((X.shape[0],(y_target != y_pred).sum()/X.shape[0]*100)) + "%")
