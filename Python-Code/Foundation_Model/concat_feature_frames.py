from pathlib import Path
import pandas as pd


def concat_feature_frames(meta_data,feature_vector,orientation):
    meta_data = meta_data[['Orientation','filename','diagnosis','patient_id']]
    meta_data = meta_data.loc[meta_data['Orientation'] == orientation]
    Whole_data_Frame = feature_vector.merge(meta_data,how='left',on='filename')

    return Whole_data_Frame