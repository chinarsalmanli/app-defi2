#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split




path = "/Users/jgodet/Seafile/MaBibliotheque/Enseignements/Ens2022-23/IDS_Apps/defi2/data/dataDefi2.csv.gz"
df = pd.read_csv(path , compression="gzip", sep=";")




features, targets = df.loc[:, df.columns != "hospital_death"], df.loc[:, df.columns == "hospital_death"]




train_features, test_features, train_targets, test_targets = train_test_split(
        features, targets,
        train_size=0.9,
        test_size=0.1,
        random_state=42,
        shuffle = True,
        stratify=targets
    )




print('Dims Train :'+str(train_features.shape))
print('Dims Train target :'+str(train_targets.shape))


print('Dims Test :'+str(test_features.shape))
print('Dims Test target :'+str(test_targets.shape))





