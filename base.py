# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

import os
import os.path

## Replace ',' with '.'
# f = open('./dataDefi2.csv')
# content = f.read()
# f.close()
# t = content.replace(",",".")
# with open("datanew.csv", "w") as f1:
#     f1.write(t)

## v0.1, remove id and icu_stay_type = transfer
# df = pd.read_csv('./datanew.csv', sep=";")
# print(df.columns)
# delist = ['encounter_id', 'patient_id', 'hospital_id', 'icu_id']
# df.drop(delist, axis=1, inplace=True)
# for index, row in df.iterrows():
#     icutype = row['icu_stay_type']
#     if icutype.find('admit') == -1:
#         df.drop(index=index, inplace=True)
#     df = df
# df.to_csv('./dfv01.csv')

df = pd.read_csv('./dfv01.csv', index_col=0)



# df = pd.read_csv('./dataDefi2.csv', sep=";")
# type(df.loc[:, 'bmi'][0], )
# for i in df.columns.tolist():
#     if type(df.loc[:, i][0]) == 'str':
#         if df.loc[:, i][0].find(',') != -1:
#             temp = np.array(df)[:, i].tolist()
#             mean = np.nanmean(temp)
#             df[np.argwhere(np.isnan(df[:, i].T)), i] = mean
#             df[i] = df[i].str.replace(r'\b(\d+),(\d+)', r'\1.\2')

# features, targets = df.loc[:, df.columns != "hospital_death"], df.loc[:, df.columns == "hospital_death"]
#
# train_features, test_features, train_targets, test_targets = train_test_split(features, targets,
#                                                                               train_size=0.8,
#                                                                               test_size=0.2,
#                                                                               random_state=42,
#                                                                               shuffle = True,
#                                                                               stratify=targets
# )

# ###### Decision Tree ######
# model = tree.DecisionTreeClassifier(criterion='gini') # algorithm as gini or entropy
#
# # model = tree.DecisionTreeRegressor() for regression
#
# # Train the model using the training sets and check score
# model.fit(X, y)
# model.score(X, y)
#
# #Predict Output
# predicted= model.predict(x_test)

###### KNN ######
# best_p = "" # 明科夫斯基的最佳参数
# best_score = 0.0
# best_k = -1
# for k in range(1, 11):
#     for p in range(1, 6):
#         knn_clf = KNeighborsClassifier(n_neighbors=k, weights="distance", p=p)
#         knn_clf.fit(train_features, train_targets)
#         score = knn_clf.score(test_features, test_targets)
#         if score > best_score:
#             best_p = p
#             best_k = k
#             best_score = score
# print("best_p = ", best_p)
# print("best_k = ", best_k)
# print("best_score = ", best_score) # 如果过大或者过小就应该适当扩大k的范围