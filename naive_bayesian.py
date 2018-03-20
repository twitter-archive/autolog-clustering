import pandas as pd
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_json("1500.json")
# ma = np.array(data.as_matrix())
# y_target = ma[:,0]
# X = ma[:,[1,10]]
X = data[['facet_1', 'facet_10', 'facet_2', 'facet_3', 'facet_4', 'facet_5', 'facet_6', 'facet_7', 'facet_8', 'facet_9']]
y_target = data['color']
y_train = set(y_target)


print("# of unique rows: %d" % X.drop_duplicates().shape[0])

gnb = GaussianNB()
y_pred = gnb.fit(X, y_target).predict(X)
y_prob = gnb.predict_proba(X)

print("Percentage of mislabeled points out of a total %d points : %d" % ((X.shape[0],(y_target != y_pred).sum()/X.shape[0]*100)) + "%")

# print((y_target != y_pred).sum())
# print(y_pred)
# print(y_prob)
counter = 0
max_probabilities = []
for entry in y_prob:
    print("Entry", counter)
    max = 0
    for probability in entry:
        print(probability)
        if(probability > max):
            max = probability
    max_probabilities.append(max)
    counter+=1

print(max_probabilities)

bad = 0
good = 0
for probability in max_probabilities:
    if(probability >= 0.5):
        good+=1
    else:
        bad+=1

print(y_train)
print("Number of good predictions:", good)
print("Number of bad predictions:", bad)
plt.scatter(max_probabilities, max_probabilities)

plt.show()
