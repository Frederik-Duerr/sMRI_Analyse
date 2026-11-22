from typing import Any

import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
class MRIDataset(Dataset):

    def __init__(self, X, y, groups):
        self.X = X
        self.y = y
        self.groups = groups

        self.patient_groups = X.groupby(groups)
        self.patient_labels = y.groupby(groups)
        self.patient_ids = groups.unique()

    def __len__(self): # This return the Number of patients we have in the group
        return self.patient_groups.ngroups

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        patient_embeddings = self.patient_groups.get_group(patient_id)
        patient_label = self.patient_labels.get_group(patient_id).iloc[0]

        row_ids = torch.tensor(patient_embeddings.index.to_numpy(), dtype=torch.long)   # NEU

        patient_embeddings = torch.tensor(
            patient_embeddings.to_numpy(),
            dtype=torch.float32
        )
        patient_label = torch.tensor(patient_label, dtype=torch.long)

        return patient_embeddings, patient_label, row_ids 
    
class AttentionAggregation(nn.Module):
    # input_dem = Number of features per slice
    # hidden_dim = Internal Dimesnsios, of the attention Network
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        print("Attention input_dim:", input_dim, type(input_dim))
        print("Attention hidden_dim:", hidden_dim, type(hidden_dim))
        self.attention = nn.Sequential(
            nn.Linear(input_dim,hidden_dim), # Learns a linear trasformation
            nn.Tanh(), # NonLinear Activationfunction
            nn.Linear(hidden_dim,1) # Creates a score for each slice
        )
        
    def forward(self, x):
        # Calculating the attention Score
        attention_score = self.attention(x)
        
        # Normaalizing the attention score
        attention_weigts = torch.softmax(attention_score,dim=1)
        
        #Weighted aggregation of sclices embeddings
        patient_embedding = torch.sum(
            attention_weigts * x,
            dim=1
        )
        return patient_embedding, attention_weigts
    
class MLPClassifiere(nn.Module):
    def __init__(self,hiddenlayer1,hiddenlayer2,hiddenlayer3,inputlayer):
        super().__init__()
        self.function = nn.Sequential(
            nn.Flatten(),
            nn.Linear(inputlayer,hiddenlayer1),
            nn.ReLU(),
            nn.Linear(hiddenlayer1,hiddenlayer2),
            
            nn.ReLU(),
            nn.Linear(hiddenlayer2,hiddenlayer3),
            
            nn.ReLU(),
            nn.Linear(hiddenlayer3,6),

        )
    def forward(self,x):
        
        function_score = self.function(x)
        
        return function_score
        
class Alzheimer(nn.Module):
    def __init__(self, input_dim, hidden_dim, hiddenlayer1,hiddenlayer2,hiddenlayer3,inputlayer ):
        super().__init__()
        self.attention = AttentionAggregation(input_dim, hidden_dim)
        self.model = MLPClassifiere(hiddenlayer1,hiddenlayer2,hiddenlayer3,inputlayer)
    
    def forward(self, x):
        patient_embeddings, patients_weight = self.attention(x)
        results = self.model(patient_embeddings)
        return results, patients_weight