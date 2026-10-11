import pandas as pd
from sklearn.feature_selection import chi2, SelectPercentile, f_classif, SelectKBest
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import numpy as np
axial = pd.read_csv('C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/axial_features_organized_with_Target.csv')
sagittal = pd.read_csv('C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/sagittal_features_organized_with_Target.csv')
coronal = pd.read_csv('C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/coronal_features_organized_with_Target.csv')


Targets_axial = axial.drop('Target',axis=1)
print(Targets_axial.columns)
Lable_axial = axial['Target']

''' Feature Selection'''
k_list = np.linspace(0,100,11,dtype=int)
print(k_list)
list_of_techniques = [chi2,f_classif]


def feature_selection(X_train,y_train,list_of_k, technique):
    comparision = {

    }

    for k in list_of_k:
        k_s = { 

        }
        for t in technique:

            new_data = SelectKBest(t,k=k).fit_transform(X_train,y_train)
            k_s.update({k:str(t)})
            k_s[str(t)] = new_data
        comparision[str(k)] = k_s

    return comparision

X_train,X_test,y_train,y_test = train_test_split(axial.drop('Target',axis=1),axial['Target'],test_size=0.2,random_state=45,stratify=axial['Target'])    
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(X_train_scaled)
feature_selection_dict = feature_selection(X_train_scaled,y_train,k_list,list_of_techniques)  

print(feature_selection_dict.items())
# pipeline = Pipeline([('Scaler',Scaler),('RandomForrest',RandomForrest())])

