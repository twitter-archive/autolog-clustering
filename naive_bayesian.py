# from sklearn_pandas import DataFrameMapper, cross_val_score
# import pandas as pd
# import numpy as np
# import sklearn.preprocessing, sklearn.decomposition, sklearn.linear_model, sklearn.pipeline, sklearn.metrics
# # from sklearn.feature_extraction.text import CountVectorizer

# from sklearn.cluster import AffinityPropagation
# from sklearn import metrics
# from sklearn.datasets.samples_generator import make_blobs
# import matplotlib.pyplot as plt

# from sklearn.naive_bayes import GaussianNB


# mock_data = pd.read_json("MOCK_DATA_HF1300.json")
# # print(list(mock_data))
# # print(mock_data["id"])


# # dataset = {}
# # names = []
# # for column_header in list(mock_data)[0:10]:
# #     dataset[column_header] = []
# #     for id in mock_data[column_header]:
# #         dataset[column_header].append(id)

# # print(dataset)
# # print(list(mock_data))

# mock_data = np.array(mock_data.as_matrix())
# X = mock_data[:, [1,11]]
# y_target = mock_data[:,0]
# # print(y_target)
# # print(y_target)

# # km = sklearn.cluster.KMeans(n_clusters=4)

# # y_pred = km.fit(X).predict(X)
# # labels = km.labels_


# # results.plot(kind="scatter")
# # plt.show()

# # values = []
# # for columns in dataset.keys():
# #     values.append(dataset[columns])
# # print(values)


# gnb = GaussianNB()
# gnb = gnb.fit(X, y_target)
# y_pred = gnb.predict(X)
# y_prob = gnb.predict_proba(X)
# # print(y_pred)

# # print("Number of mislabeled points out of a total %d points : %d" % (X.shape[0], (y_target != y_pred).sum()))
# arr_true_0_indices = (y_test == 0.0)
# arr_true_1_indices = (y_test == 1.0)

# arr_pred_0 = prediction[arr_true_0_indices]
# arr_pred_1 = prediction[arr_true_1_indices]

# plt.hist(arr_pred_0, bins=40, label='True class 0', normed=True, histtype='step')
# plt.hist(arr_pred_1, bins=40, label='True class 1', normed=True, histtype='step')
# plt.xlabel('Network output')
# plt.ylabel('Arbitrary units / probability')
# plt.legend(loc='best')
# plt.show()
# # plt.scatter(y_pred, y_target)
# plt.scatter(y_prob.shape[0], y_target)

# plt.show()

import numpy as np
import matplotlib.pyplot as plt

x = np.random.randn(10)
y = np.random.randn(10)
Cluster = np.array([0, 1, 1, 1, 3, 2, 2, 3, 0, 2])    # Labels of cluster 0 to 3
centers = np.random.randn(4, 2) 

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x,y,c=Cluster,s=50)
for i,j in centers:
    ax.scatter(i,j,s=50,c='red',marker='+')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(scatter)

plt.show()

# take and create a new dataset, facet 1 = facet 2 = facet 3 = facet 10 = 1
# 
