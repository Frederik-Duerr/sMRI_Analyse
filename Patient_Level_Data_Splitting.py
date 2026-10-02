import pandas as pd

def patient_Table(dataFrame):
    id_diag = dataFrame[['patient_id','diagnosis']] # Now we have a smaller DataFrame with just the important values
    new_df = id_diag.groupby('patient_id')['diagnosis'].unique() # We group by the patient id and secondly by the diagnosis, the last comands selects only unique items. We end up with our 120 unique patient ids and theris diagnosis
    return new_df
