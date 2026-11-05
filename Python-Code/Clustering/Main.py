import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib. pyplot as plt
import numpy as np
# Loading in data

axial_dino = pd.read_parquet('C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/Axial_Features.parquet')
sagittal_dino = pd.read_parquet('C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/Sagittal_Features.parquet')
coronal_dino = pd.read_parquet('C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/Coronal_Features.parquet')

# concatted_dino = pd.read_excel("C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Python-Code/Foundation_Model/Concated_Features_With_lable.xlsx")
print()
# X = concatted_dino.drop()

# ============================================================
# 5. LOAD Matlab extracted Features
# ============================================================
# Set these paths to the directory where your features extraction data is stored
axial = pd.read_csv('C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/axial_features_organized_with_Target.csv')
sagittal = pd.read_csv('C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/sagittal_features_organized_with_Target.csv')
coronal = pd.read_csv('C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/coronal_features_organized_with_Target.csv')

print(axial.info())
merged_MATLAB = pd.concat([axial,sagittal,coronal],axis=0,join='inner',ignore_index=True)
# merged_MATLAB = merged_MATLAB.merge(sagittal,how='inner', on= 'PatientID')
print(merged_MATLAB.columns)
X  = merged_MATLAB.drop(['PatientID','Group','Target'],axis=1)
y = merged_MATLAB['Target']
print(X.columns)

k = 6
OMP_NUM_THREADS=2
pca = PCA(n_components=3)
reduced_data = pca.fit_transform(X)

kmeans = KMeans(n_clusters=k,random_state=42)
y_pred = kmeans.fit_predict(X)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y_pred, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, label='Centroids')
plt.legend()
plt.show()

#Getting unique labels
 
u_labels = np.unique(X)
 
#plotting the results:
 
for i in u_labels:
    plt.scatter(X[y == i , 0] , X[y == i , 1] , label = i)
    plt.legend()
    plt.show()