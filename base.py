# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
from sklearn.decomposition import PCA
from dask.distributed import Client
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

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
all_features_pca = pca.transform(features)

## v1.2 DASK
def daskrun(MLmethod, train, test):
    daskop = Client(n_workers=4)
    daskop
    with joblib.parallel_backend('dask'):
        MLmethod.fit(train, np.ravel(test))
    print(MLmethod.score(test_features_pca, test_targets))

## v1.3.1 undersampling
num_train_target = train_targets.sum()
# print('Number of samples:{}'.format(num_train_target))

undersample = RandomUnderSampler(sampling_strategy=0.2, random_state=42)
train_features_pca_under, train_targets_under = undersample.fit_resample(train_features_pca, train_targets)
# print(train_features_pca_under.shape)

## v1.3.2 oversampling
oversample = SMOTE(sampling_strategy=0.2, random_state=42)
train_features_pca_over, train_targets_over = oversample.fit_resample(train_features_pca, train_targets)
# print(train_features_pca_over.shape)

## v2.0 models
if __name__ == '__main__':
    # daskop = Client(n_workers = 4)
    # daskop


    # ###### Decision Tree ######  0.91887 0.92028
    DT = DecisionTreeClassifier(max_depth=9,
                                min_samples_leaf=19,
                                min_samples_split=3,
                                splitter='random',
                                criterion='gini')

    # [0.28002739 0.27926924 0.27834374 0.27700758 0.27915865] 9
    # [0.28468702 0.28147737 0.28839521 0.28681552 0.28596946]

    # [0.27497947 0.27409688 0.27373545 0.27246006 0.27298757] 15
    # [0.28543075 0.28447417 0.29173661 0.29339304 0.28373653]

    # [0.38533483 0.40506021 0.36550308 0.40397269 0.40049536]
    # [0.32124874 0.33217305 0.307      0.29382958 0.31039679]

    tempdt =DecisionTreeClassifier(max_depth=18,
                                   min_samples_leaf=17,
                                   min_samples_split=3,
                                   splitter='random',
                                   criterion='gini')
    # daskrun(tempdt, train_features_pca_under, train_targets_under)
    # 0.4545057995634662
    # paras = {
    #     #'criterion': ('gini', 'entropy'),
    #     #'splitter': ('best', 'random'),
    #     'max_depth': (list(range(1,20))),
    #     'min_samples_split': [2, 3, 4],
    #     'min_samples_leaf': list(range(1, 20))
    # }
    # grid_dt = GridSearchCV(tempdt, paras, cv=3, scoring='f1', n_jobs=-1)
    # # grid_dt = RandomizedSearchCV(DT, paras, cv=3, scoring='f1', n_iter=300, n_jobs=-1)
    # grid_dt.fit(train_features_pca_under, np.ravel(train_targets_under))
    # best_dt = grid_dt.best_estimator_
    # print(best_dt)
    # print(grid_dt.best_score_)
    # with joblib.parallel_backend('dask'):
    #     DT.fit(train_features_pca_under, np.ravel(train_targets_under))
    # print(DT.score(test_features_pca, test_targets))

    ###### KNN ######  0.9239
    knn = KNeighborsClassifier(n_neighbors=9, weights="uniform", p=4)

    # with joblib.parallel_backend('dask'):
    #     knn.fit(train_features_pca, train_targets)
    # print(knn.score(test_features_pca, test_targets))

    # [0.26989325 0.27036977 0.26851298 0.26828729 0.26899194]
    # [0.27692137 0.27724928 0.28137828 0.28416321 0.27780335]

    # Undersampling
    tempknn = KNeighborsClassifier(n_neighbors=3, weights="uniform", p=2)
    # daskrun(tempknn, train_features_pca_under, train_targets_under)
    # [0.53387308 0.53406254 0.52939046 0.54362496 0.54450758] p=2
    # [0.33065327 0.32179226 0.314      0.28658848 0.29087158]

    # [0.53571429 0.53393558 0.53216304 0.5392378  0.53315902] p=4
    # [0.31313131 0.30463576 0.30441249 0.27526663 0.28012358]

    # [0.53882688 0.53663178 0.53021348 0.5466031  0.53166667] p=6
    # [0.30463576 0.29417773 0.3047996  0.28006088 0.29028926]
    ##############################GridSearch##############################
    # paras = {
    #     'n_neighbors': (list(range(1,11))),
    #     'p': list(range(1, 6))
    # }
    # grid_dt = GridSearchCV(tempknn, paras, cv=3, scoring='f1', n_jobs=-1)
    # grid_dt.fit(train_features_pca_under, np.ravel(train_targets_under))
    # best_dt = grid_dt.best_estimator_
    # print(best_dt)
    # print(grid_dt.best_score_)
    ######################################################################

    ###### Random Forest ###### 0.925 0.927
    RF = RandomForestClassifier(n_estimators=1500,
                                max_features='auto',
                                bootstrap=False,
                                max_depth=15,
                                random_state=42)
    # with joblib.parallel_backend('dask'):
    #     RF.fit(train_features_pca_under, np.ravel(train_targets_under))
    # print(RF.score(test_features_pca, test_targets))

    # [0.25114642 0.25111627 0.24963249 0.24856861 0.24938972] 10
    # [0.27670254 0.27184386 0.27900008 0.28094737 0.27671093]

    # [0.20205957 0.20190962 0.2007799  0.20149515 0.20059125] 15
    # [0.27262263 0.26848072 0.27506415 0.27703911 0.27407133]

    # Undersampling
    temprf = RandomForestClassifier(n_estimators=1500,
                                    max_features='auto')

    # [0.69286853 0.69402137 0.69402299 0.69356498 0.69762149]
    # [0.31414198 0.32309443 0.31486561 0.30346344 0.30803571]

    ##############################GridSearch##############################
    # paras = {
    #     # 'n_estimators': [100, 500, 1000, 1500],
    #     #'max_features': ['auto', 'sqrt', 'log2'],
    #     'max_depth': [5, 10, 15],
    #     'min_samples_split': [2, 5, 10],
    #     # 'min_samples_leaf': [1, 2, 4, 10],
    #     # 'bootstrap': [True, False]
    # }
    # grid_rf = GridSearchCV(temprf, paras, scoring='f1', cv=3, verbose=2, n_jobs=-1)
    # grid_rf.fit(train_features_pca_under, np.ravel(train_targets_under))
    # best_rf = grid_rf.best_estimator_
    # print(best_rf)
    # print(grid_rf.best_score_)
    ######################################################################

    ###### SVM ######
    def PolynomialSVC(degree, C=1.0):
        return Pipeline([
            ("poly", PolynomialFeatures(degree=degree)),
            ("linearSVC", LinearSVC(C=C))
        ])


    poly_svc = PolynomialSVC(degree=3) # 0.8617723665879218
    kernel_poly_svc = SVC(kernel="poly", degree=3, C=1.0) # 0.9277363862135805
    # [0.47889838 0.47153434 0.47305157 0.4833206  0.48824273]
    # [0.36951983 0.36752137 0.37423935 0.35592344 0.35905512]
    rbf_svc = SVC(kernel='rbf', gamma=0.1) # [0.1, 0.5, 1]
    # daskrun(kernel_poly_svc, train_features_pca_under, train_targets_under)

    ###### voting ######
    voting = VotingClassifier(estimators=[
        ('DecisionTree', tempdt),
        ('SVM', SVC(kernel="poly", degree=3, C=1.0, probability=True)),
        ('KNN', tempknn)],
        voting='soft')

    # daskrun(voting, train_features_pca_under, train_targets_under)

    # [0.54848065 0.53959018 0.54336441 0.5528209  0.5572002]
    # [0.31892195 0.30389908 0.30633609 0.2909699  0.30498866]

    # daskrun(voting)

## v3.0 Cross Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    stracv = StratifiedKFold(n_splits=5)
    result_f = cross_validate(voting,
                              all_features_pca, np.ravel(targets),
                              cv=stracv,
                              scoring="f1",
                              return_train_score=True,
                              verbose=True,
                              n_jobs=-1,
                              )
    # # trainRMSE_f = abs(result_f["train_score"]) ** 0.5
    # # testRMSE_f = abs(result_f["test_score"]) ** 0.5
    # # print(trainRMSE_f)
    # # print(testRMSE_f)
    trainf1 = result_f["train_score"]
    testf1 = result_f["test_score"]
    print(trainf1)
    print(testf1)





