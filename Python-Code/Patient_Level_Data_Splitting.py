import pandas as pd
import numpy as np
def patient_Table(dataFrame):
    id_diag = dataFrame[['patient_id','lable']] # Now we have a smaller DataFrame with just the important values
    new_df = id_diag.groupby('patient_id')['lable'].unique() # We group by the patient id and secondly by the diagnosis, the last comands selects only unique items. We end up with our 120 unique patient ids and theris diagnosis
    new_df = pd.DataFrame(new_df)
    new_df['lable'] = new_df['lable'].apply(lambda x: x[0]) # With the unique methode we returned a list, and here we just take the first item of the list
    new_df = new_df.reset_index()
    return new_df
