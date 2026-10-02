import pandas as pd

def patient_Table(dataFrame):
    id_Diag = dataFrame[['patient_id','diagnosis']]
    print(id_Diag.head())
    id_set = set()
    for id in id_Diag['patient_id']:
        id_set.add(id)
    print(id_set)