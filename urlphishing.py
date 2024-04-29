 pip install pandas numpy seaborn matplotlib pycaret



#IMPORTING ALL NECESSARY LIBRARIES & PACKAGES

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from pycaret.classification import *


#IMPORTING THE DATASET AND SHOWING IT

df=pd.read_csv('/content/dataset_full.csv')
df


#SUMMARY STATISTICS OF THE DATASET

print(df.describe())


#VISUALIZING THE DISTRIBUTION OF THE DATASET

sns.countplot(x='phishing', data=df)
plt.show()


cols_to_drop = ['url_google_index',
                  'domain_google_index',
                  'qty_vowels_domain',
                  'server_client_domain',
                  'tld_present_params',
                  'time_response',
                  'domain_spf',
                  'qty_ip_resolved',
                  'qty_nameservers',
                  'qty_mx_servers',
                  'ttl_hostname',
                  'url_shortened']
df=df.drop(cols_to_drop, axis=1)
df


#PERFORMING FEATURE ENGINEERING ON DATA AND SHOWING IT
rows, columns = df.shape

original_features = list(df.columns)

dataset_array = np.array(df)

features_indices = []
attributes = ['url', 'domain', 'directory', 'file', 'params']

new_dataset = {}

for index, name in enumerate(original_features):
    if 'qty' in name and name.split('_')[-1] in attributes:
        features_indices.append([index, name.split('_')[-1]])
    else:
        new_dataset[name] = dataset_array[:, index]


for index, attribute in features_indices:
  if attribute == 'domain':
    if f"qty_char_{attribute}" not in new_dataset.keys():
        new_dataset[f"qty_char_{attribute}"] = np.zeros(rows)

    new_dataset[f"qty_char_{attribute}"] += dataset_array[:,index]

df1 = pd.DataFrame(new_dataset).astype(int)
df1[df1<-1] = -1
df1


# prompt: Using dataframe df1: domain_in_ip

df1.query("domain_in_ip == 1").head()




#PLOTTING A CORRELATION MATRIX TO IDENTIFY ANY STRONG CORRELATIONS

corr = df1.corr()
plt.figure(figsize=(100, 100))
sns.heatmap(corr, annot=True)
plt.show()


#SUMMARY STATISTICS OF THE DATASET

print(df1.describe())



#SETTING UP THE DATA FOR MODELLING

setup(data=df1, target='phishing')



#COMPARING AND SELECTING THE BEST DATA

best_model = compare_models()



#TUNING THE HYPERPARAMETERS OF THE BEST PERFORMING MODEL

tuned_model = tune_model(best_model, n_iter=1, optimize='F1')



pip install shap



#MAKING PREDICTIONS ON NEW DATA

predictions = predict_model(tuned_model, data=df1)



