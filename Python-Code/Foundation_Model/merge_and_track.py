"""
Hilfsfunktionen, damit deine eigenen Klassen (MRIDataset, AttentionAggregation,
MLPClassifiere, Alzheimer) alle drei Orientierungen in EINEM Bag benutzen koennen
und ID, Diagnose und Orientierung mitgespeichert werden.

Deine Klassen bleiben unveraendert. Diese Datei enthaelt nur:
  1. merge_orientations()   - klebt die drei Orientierungen untereinander
  2. MRIDatasetTracked      - dein MRIDataset, gibt zusaetzlich die Patienten-ID zurueck
  3. collect_predictions()  - speichert Vorhersage + Attention-Gewicht pro Slice
"""
import pandas as pd
import torch
from torch.utils.data import DataLoader

from MultiinstanceLearning import MRIDataset   # deine Klasse


# ======================================================================
# 1) Die drei Orientierungen zu einem Datensatz zusammenkleben
# ======================================================================
def merge_orientations(Test_train_by_orientation):
    """Eingabe : dein Dictionary {'axial': {...}, 'coronal': {...}, 'sagittal': {...}}
    Ausgabe: ein Dictionary mit EINEM Eintrag 'merged' im gleichen Format wie bisher,
             plus 'meta_train' / 'meta_test'. Das sind Tabellen, die Zeile fuer Zeile zu
             X_train / X_test passen (patient_id, label, orientation, slice_idx, slice_pos).
    """
    merged_data = {}

    for split in ["train", "test"]:
        X_parts, y_parts, id_parts, meta_parts = [], [], [], []

        for orientation, Data in Test_train_by_orientation.items():
            X = Data[f"X_{split}"].reset_index(drop=True)
            X.columns = range(X.shape[1])          # gleiche Spaltennamen in allen Orientierungen
            y = pd.Series(Data[f"y_{split}"].to_numpy())
            ids = pd.Series(Data[f"patient_ids_{split}"].to_numpy()).astype(str)

            # Metadaten: gleiche Zeilenreihenfolge wie X
            meta = pd.DataFrame({"patient_id": ids, "label": y, "orientation": orientation})
            meta["slice_idx"] = meta.groupby("patient_id").cumcount()   # Nummer des Slices
            n_slices = meta.groupby("patient_id")["slice_idx"].transform("size")
            meta["slice_pos"] = meta["slice_idx"] / (n_slices - 1).clip(lower=1)   # 0 bis 1

            X_parts.append(X)
            y_parts.append(y)
            id_parts.append(ids)
            meta_parts.append(meta)

        # untereinander kleben (axial, dann coronal, dann sagittal)
        merged_data[f"X_{split}"] = pd.concat(X_parts, ignore_index=True)
        merged_data[f"y_{split}"] = pd.concat(y_parts, ignore_index=True)
        merged_data[f"patient_ids_{split}"] = pd.concat(id_parts, ignore_index=True)
        merged_data[f"meta_{split}"] = pd.concat(meta_parts, ignore_index=True)

        # Kontrollen
        rows_per_patient = merged_data[f"patient_ids_{split}"].value_counts()
        print(f"{split}: {len(rows_per_patient)} Patienten, "
              f"Zeilen pro Patient: {sorted(rows_per_patient.unique())}")
        if len(rows_per_patient.unique()) != 1:
            print("WARNUNG: Patienten haben unterschiedlich viele Slices. "
                  "Der normale DataLoader braucht gleich grosse Bags.")
        n_labels = merged_data[f"meta_{split}"].groupby("patient_id")["label"].nunique()
        assert (n_labels == 1).all(), "Ein Patient hat mehr als eine Diagnose!"

    return {"merged": merged_data}


# ======================================================================
# 2) Dein MRIDataset, aber mit Patienten-ID als drittem Rueckgabewert
# ======================================================================
class MRIDatasetTracked(MRIDataset):
    def __getitem__(self, idx):
        patient_embeddings, patient_label = super().__getitem__(idx)
        return patient_embeddings, patient_label, str(self.patient_ids[idx])


# ======================================================================
# 3) Vorhersage und Attention pro Slice einsammeln
# ======================================================================
def collect_predictions(model, X_val, y_val, groups_val, meta_val):
    """Gibt zwei Tabellen zurueck:
    pred_df: eine Zeile pro Patient  (patient_id, label, pred, p0 ... p5)
    attn_df: eine Zeile pro Slice    (patient_id, label, orientation, slice_idx, slice_pos, attention)

    Die Reihenfolge der Slices im Bag ist die Reihenfolge der Zeilen des Patienten in X.
    Dieselbe Reihenfolge hat meta_val, deshalb passt das k-te Gewicht zur k-ten Zeile von meta_val.
    """
    dataset = MRIDatasetTracked(X_val, y_val, groups_val)
    loader = DataLoader(dataset, batch_size=1)
    meta_lookup = {pid: g for pid, g in meta_val.groupby("patient_id", sort=False)}

    model.eval()
    pred_rows, attn_parts = [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_id in loader:
            scores, weights = model(batch_x)                  # weights: [1, Anzahl Slices, 1]
            probs = torch.softmax(scores, dim=1)[0]
            pid = batch_id[0]

            row = {"patient_id": pid, "label": batch_y[0].item(), "pred": probs.argmax().item()}
            row.update({f"p{c}": probs[c].item() for c in range(len(probs))})
            pred_rows.append(row)

            slices = meta_lookup[pid].copy()                  # Slices dieses Patienten, gleiche Reihenfolge
            att = weights[0, :, 0].numpy()
            assert len(slices) == len(att), "Anzahl Slices passt nicht zur Anzahl Attention-Gewichte"
            slices["attention"] = att
            attn_parts.append(slices)

    return pd.DataFrame(pred_rows), pd.concat(attn_parts, ignore_index=True)
