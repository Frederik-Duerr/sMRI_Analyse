import pandas as pd

def patient_Table(dataFrame):
    id = dataFrame['patient_id','diagnosis']
    id = id.unique()
    Patient_table = dataFrame[id]
    return Patient_table