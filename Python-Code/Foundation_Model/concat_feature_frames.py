from pathlib import Path
import pandas as pd


def concat_feature_frames(meta_data,feature_vector,orientation):
    meta_data = meta_data.loc[meta_data['Orientation'] == orientation,'filename']
    Whole_data_Frame = feature_vector.merge(meta_data,how='outer',on='filename')

    return Whole_data_Frame