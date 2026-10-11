import pandas as pd

axial = pd.read_csv('C:/Users/Johan/OneDrive - ucp.pt/Desktop/Privat/Portugal/UCP/Semester 3/Biomedical Project/MRI Project/Features/axial_features_organized_with_Target.csv')
print(axial.head())
print(axial.info)

print(axial.columns)
print(f'Shape: {axial.shape}')