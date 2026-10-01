import pandas as pd

def patient_Table(dataFrame):
    id = dataFrame['patient_id']
    id = id.unique()
    Patient_table = dataFrame[id,'diagnosis']
    return Patient_table