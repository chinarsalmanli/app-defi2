# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
from sklearn.decomposition import PCA
from dask.distributed import Client
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import os
import os.path

df = pd.read_csv('dataDefi2.csv', sep=';')

## v0.1 Remove id
delist = ['encounter_id', 'patient_id', 'hospital_id', 'icu_id', 'apache_3j_bodysystem', 'apache_2_bodysystem']
df.drop(delist, axis=1, inplace=True)

## v0.2 Replace ',' with '.'
#print(df.columns)
repvir = ['bmi', 'height', 'pre_icu_los_days', 'weight', 'apache_3j_diagnosis', 'temp_apache', 'd1_temp_max', 'd1_temp_min', 'd1_potassium_max', 'd1_potassium_min', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']
for i in repvir:
    # if str(df[i]).find(',') != -1:
    df[i] = df[i].str.replace(',', '.').astype('float')

## v0.3 Clean and standardlize
dummcol = ['ethnicity', 'icu_admit_source', 'icu_stay_type', 'icu_type']
df = pd.get_dummies(data=df, columns=dummcol)
df['gender'] = df['gender'].apply(lambda x: 0 if x=='F' else 1)
# for j in df.columns.tolist():
#     df[j].replace(to_replace=[None],value=np.nan,inplace=True)
# df = df.groupby(df.columns, axis = 1).transform(lambda x: x.fillna(x.mean()))
for j in df.columns.tolist():
    df[j] = pd.to_numeric(df[j], errors='coerce')
for column in list(df.columns[df.isnull().sum() > 0]):
    mean_val = df[column].mean()
    df[column].fillna(mean_val, inplace=True)

# print(df.columns)

std = StandardScaler()
std = std.fit_transform(df.loc[:, df.columns != 'hospital_death'])
attricol = ['age', 'bmi', 'elective_surgery', 'gender', 'height',
       'pre_icu_los_days', 'weight', 'apache_2_diagnosis',
       'apache_3j_diagnosis', 'apache_post_operative', 'arf_apache',
       'gcs_eyes_apache', 'gcs_motor_apache', 'gcs_unable_apache',
       'gcs_verbal_apache', 'heart_rate_apache', 'intubated_apache',
       'map_apache', 'resprate_apache', 'temp_apache', 'ventilated_apache',
       'd1_diasbp_max', 'd1_diasbp_min', 'd1_diasbp_noninvasive_max',
       'd1_diasbp_noninvasive_min', 'd1_heartrate_max', 'd1_heartrate_min',
       'd1_mbp_max', 'd1_mbp_min', 'd1_mbp_noninvasive_max',
       'd1_mbp_noninvasive_min', 'd1_resprate_max', 'd1_resprate_min',
       'd1_spo2_max', 'd1_spo2_min', 'd1_sysbp_max', 'd1_sysbp_min',
       'd1_sysbp_noninvasive_max', 'd1_sysbp_noninvasive_min', 'd1_temp_max',
       'd1_temp_min', 'h1_diasbp_max', 'h1_diasbp_min',
       'h1_diasbp_noninvasive_max', 'h1_diasbp_noninvasive_min',
       'h1_heartrate_max', 'h1_heartrate_min', 'h1_mbp_max', 'h1_mbp_min',
       'h1_mbp_noninvasive_max', 'h1_mbp_noninvasive_min', 'h1_resprate_max',
       'h1_resprate_min', 'h1_spo2_max', 'h1_spo2_min', 'h1_sysbp_max',
       'h1_sysbp_min', 'h1_sysbp_noninvasive_max', 'h1_sysbp_noninvasive_min',
       'd1_glucose_max', 'd1_glucose_min', 'd1_potassium_max',
       'd1_potassium_min', 'apache_4a_hospital_death_prob',
       'apache_4a_icu_death_prob', 'aids', 'cirrhosis', 'diabetes_mellitus',
       'hepatic_failure', 'immunosuppression', 'leukemia', 'lymphoma',
       'solid_tumor_with_metastasis',
       'ethnicity_African American', 'ethnicity_Asian', 'ethnicity_Caucasian',
       'ethnicity_Hispanic', 'ethnicity_Native American',
       'ethnicity_Other/Unknown', 'icu_admit_source_Accident & Emergency',
       'icu_admit_source_Floor', 'icu_admit_source_Operating Room / Recovery',
       'icu_admit_source_Other Hospital', 'icu_admit_source_Other ICU',
       'icu_stay_type_admit', 'icu_stay_type_readmit',
       'icu_stay_type_transfer', 'icu_type_CCU-CTICU', 'icu_type_CSICU',
       'icu_type_CTICU', 'icu_type_Cardiac ICU', 'icu_type_MICU',
       'icu_type_Med-Surg ICU', 'icu_type_Neuro ICU', 'icu_type_SICU']
std = pd.DataFrame(data=std, columns= attricol)
atop = pd.concat([std, pd.DataFrame(columns=['hospital_death'])])
atop['hospital_death'] = df['hospital_death']

## v0.4 show attributs' importance
# atopdata, atoptarget = atop.loc[:, atop.columns != 'hospital_death'], atop.loc[:, atop.columns == 'hospital_death']
# names = std.columns
# rf=RandomForestClassifier()
# rf.fit(atopdata, atoptarget)
# atopimportance = rf.feature_importances_
# for k,l in enumerate(atopimportance):
#        print('Feature: %0d, Score: %.5f' % (k, l))
# pyplot.bar([x for x in range(len(atopimportance))], atopimportance)
# pyplot.show()
# # print("Features sorted by their score:")
# # print(sorted(zip(map(lambda x:round(x,4),rf.feature_importances_),names)))

## v0.5 remove attributs with importance<0.01
temp1 = [x for x in range(atop.shape[1])]
temp2 = [2,3,9,10,13,16,33,53]
temp3 = [x for x in range(65,95)]
tempf = []
for m in temp3:
       temp2.append(m)
# print(temp2)
for n in temp1:
       if n not in temp2:
              tempf.append(n)
#print(tempf)
atopdata = atop.iloc[:, tempf]
atoptarget = atop.loc[:, atop.columns == 'hospital_death']

# -----------------------------------------------------------------------------

## v1.0 split train set and test set
features, targets = atopdata, atoptarget

train_features, test_features, train_targets, test_targets = train_test_split(features, targets,
                                                                              train_size=0.8,
                                                                              test_size=0.2,
                                                                              random_state=42,
                                                                              shuffle = True,
                                                                              stratify=targets
)

## v1.1 PCA
pca = PCA(0.95)
pca.fit(train_features)
# print(pca.n_components_) # = 32

train_features_pca = pca.transform(train_features)
test_features_pca = pca.transform(test_features)

## v1.2 DASK
if __name__ == '__main__':
    daskop = Client(n_workers = 4)
    daskop


    # ###### Decision Tree ######  0.91887 0.92028
    DT = DecisionTreeClassifier(max_depth=9, min_samples_leaf=19, min_samples_split=3, splitter='random', criterion='gini')
    # paras = {
    #     'criterion': ('gini', 'entropy'),
    #     'splitter': ('best', 'random'),
    #     'max_depth': (list(range(1,20))),
    #     'min_samples_split': [2, 3, 4],
    #     'min_samples_leaf': list(range(1, 20))
    # }
    # grid_dt = GridSearchCV(DT, paras, cv=3, scoring='accuracy', n_jobs=-1)
    # # grid_dt = RandomizedSearchCV(DT, paras, cv=3, scoring='accuracy', n_iter=300, n_jobs=-1)
    # grid_dt.fit(train_features_pca, np.ravel(train_targets))
    # best_dt = grid_dt.best_estimator_
    # print(best_dt)
    # print(grid_dt.best_score_)
    with joblib.parallel_backend('dask'):
        DT.fit(train_features_pca, np.ravel(train_targets))
    print(DT.score(test_features_pca, test_targets))

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

    ###### KNN ######  0.9239
    # with joblib.parallel_backend('dask'):
    #     # best_p = "" # 明科夫斯基的最佳参数
    #     best_score = 0.0
    #     best_k = -1
    #     for k in range(1, 11):
    #     # for p in range(1, 6):
    #         knn_clf = KNeighborsClassifier(n_neighbors=k ,weights="uniform", p=4)
    #         knn_clf.fit(train_features_pca, train_targets)
    #         score = knn_clf.score(test_features_pca, test_targets)
    #         if score > best_score:
    #             # best_p = p
    #             best_k = k
    #             best_score = score
    #     # print("best_p = ", best_p)
    #     print("best_k = ", best_k)
    #     print("best_score = ", best_score) # 如果过大或者过小就应该适当扩大k的范围


    # knn = KNeighborsClassifier(n_neighbors=9 ,weights="uniform", p=4)
    # with joblib.parallel_backend('dask'):
    #     knn.fit(train_features_pca, train_targets)
    # print(knn.score(test_features_pca, test_targets))
