from torch.utils.data import Dataset,DataLoader

class MultiOrientationDataset(Dataset):

    def __init__(self, X_dict, y_dict, groups_dict):
        self.orientations = ['axial', 'coronal', 'sagittal']


        self.patient_ids = groups_dict['axial'].unique()


        self.patient_groups = {
            orientation: X_dict[orientation].groupby(
                groups_dict[orientation]
            )
            for orientation in self.orientations
        }

        self.patient_labels = {
            orientation: y_dict[orientation].groupby(
                groups_dict[orientation]
            )
            for orientation in self.orientations
        }
    def __getitem__(self, patient_groups):
        data = {
            
        }
        for ori, data_ in patient_groups.items():
            data[ori] = data_ 
        
        return data
    def __len__(self):
        return self.patient_groups['axial'].ngroups()