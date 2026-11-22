"""
Multi-orientation MIL for MRI classification (axial + coronal + sagittal in ONE bag).

Idea
----
* One bag = one patient. The bag contains the slices of ALL orientations.
* Every slice row keeps its metadata (patient_id, label, orientation, slice_idx, ...)
  in a table `meta` whose row i describes row i of the feature matrix `X_all`.
* The dataset hands the row numbers of each bag through the DataLoader, so every
  prediction and every attention weight can be joined back to its metadata.

Assumptions about your Data_Dictionary (same as your current pipeline)
----------------------------------------------------------------------
Data_Dictionary[orientation] = {
    "X_train": DataFrame [n_slices, D],  "y_train": Series (int 0..5, one per slice),
    "patient_ids_train": Series (one per slice),
    "X_test": ..., "y_test": ..., "patient_ids_test": ...
}
Rows of one patient inside one orientation are stored in slice order.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

N_CLASSES = 6


# ======================================================================
# 1) Merge the orientations and keep all metadata
# ======================================================================
def build_long_data(Data_Dictionary, splits=("train", "test")):
    """Stack all slices of all orientations into one table.
    splits=("train", "test") uses all patients; splits=("train",) leaves your original
    test patients out completely (keep them for a final evaluation).

    Returns
    -------
    X_all : float32 array [n_slices_total, D]
    meta  : DataFrame, row i describes X_all[i]
            columns: patient_id, label, orientation, orient_id, orig_split,
                     slice_idx, n_slices, slice_pos
    """
    orient2id = {o: i for i, o in enumerate(Data_Dictionary)}
    X_parts, meta_parts = [], []
    for orientation, Data in Data_Dictionary.items():
        for split in splits:
            X = Data[f"X_{split}"]
            meta_parts.append(pd.DataFrame({
                "patient_id": Data[f"patient_ids_{split}"].to_numpy(),
                "label": Data[f"y_{split}"].to_numpy().astype(int),
                "orientation": orientation,
                "orient_id": orient2id[orientation],
                "orig_split": split,
            }))
            X_parts.append(X.to_numpy(dtype=np.float32))

    X_all = np.vstack(X_parts)
    meta = pd.concat(meta_parts, ignore_index=True)   # RangeIndex == row number in X_all

    # position of each slice inside its (patient, orientation) stack
    meta["slice_idx"] = meta.groupby(["patient_id", "orientation"], sort=False).cumcount()
    meta["n_slices"] = (meta.groupby(["patient_id", "orientation"], sort=False)["slice_idx"]
                            .transform("size"))
    meta["slice_pos"] = meta["slice_idx"] / (meta["n_slices"] - 1).clip(lower=1)  # 0..1
    assert len(meta) == len(X_all)
    return X_all, meta


def check_data(meta):
    """Fail early on the problems that silently ruin a merged MIL setup."""
    n_lab = meta.groupby("patient_id")["label"].nunique()
    assert (n_lab == 1).all(), f"Patients with more than one label: {n_lab[n_lab > 1].index.tolist()[:10]}"

    n_or = meta.groupby("patient_id")["orientation"].nunique()
    assert (n_or == meta["orientation"].nunique()).all(), \
        f"Patients missing in some orientation: {n_or[n_or < meta['orientation'].nunique()].index.tolist()[:10]}"

    n_split = meta.groupby("patient_id")["orig_split"].nunique()
    if (n_split > 1).any():
        print(f"WARNING: for {(n_split > 1).sum()} patients your ORIGINAL train/test split differs between "
              f"orientations. This script ignores the original split and splits by patient once.")

    per_patient = meta.groupby(["patient_id", "orientation"]).size().unstack("orientation")
    print("Slices per patient and orientation (min / max):")
    print(pd.DataFrame({"min": per_patient.min(), "max": per_patient.max()}))
    patients = meta.drop_duplicates("patient_id")
    print("Patients per class:\n", patients["label"].value_counts().sort_index())


# ======================================================================
# 2) Dataset: one item = one patient, rows are carried along for tracking
# ======================================================================
class BagDataset(Dataset):
    def __init__(self, X, meta, patient_ids):
        self.X = X
        self.orient = meta["orient_id"].to_numpy()
        self.items = []
        sub = meta[meta["patient_id"].isin(set(patient_ids))]
        for pid, g in sub.groupby("patient_id", sort=False):
            self.items.append({"patient_id": pid,
                               "rows": g.index.to_numpy(),      # row numbers in X_all / meta
                               "label": int(g["label"].iloc[0])})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        r = it["rows"]
        return {"patient_id": it["patient_id"],
                "rows": torch.from_numpy(r),
                "x": torch.from_numpy(self.X[r]),
                "orient": torch.from_numpy(self.orient[r]),
                "label": it["label"]}


def collate_bags(batch):
    lengths = torch.tensor([len(b["rows"]) for b in batch])
    x = pad_sequence([b["x"] for b in batch], batch_first=True)                       # [B, N, D]
    orient = pad_sequence([b["orient"] for b in batch], batch_first=True)             # [B, N]
    rows = pad_sequence([b["rows"] for b in batch], batch_first=True, padding_value=-1)
    mask = torch.arange(x.shape[1])[None, :] < lengths[:, None]                       # True = real slice
    return {"x": x, "orient": orient, "rows": rows, "mask": mask,
            "y": torch.tensor([b["label"] for b in batch]),
            "patient_id": [b["patient_id"] for b in batch]}


def drop_slices(mask, p):
    """Slice dropout for training: randomly hide a fraction p of the real slices (>=1 stays)."""
    keep = mask & (torch.rand(mask.shape) > p)
    empty = keep.sum(dim=1) == 0
    keep[empty] = mask[empty]
    return keep


# ======================================================================
# 3) Model: gated attention + orientation embedding
# ======================================================================
class GatedAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.25):
        super().__init__()
        self.V = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        score = self.w(self.drop(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))))   # [B, N, 1]
        score = score.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(score, dim=1)                                          # over slices
        return (weights * x).sum(dim=1), weights.squeeze(-1)                           # [B, D], [B, N]


class MultiOrientationMIL(nn.Module):
    def __init__(self, input_dim=384, n_orient=3, attn_dim=128, hidden=256,
                 n_classes=N_CLASSES, dropout=0.4):
        super().__init__()
        self.orient_emb = nn.Embedding(n_orient, input_dim)   # tells the model which orientation a slice has
        nn.init.normal_(self.orient_emb.weight, std=0.02)
        self.attention = GatedAttention(input_dim, attn_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Dropout(dropout),
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x, orient, mask):
        x = x + self.orient_emb(orient)
        bag, weights = self.attention(x, mask)
        return self.classifier(bag), weights


# ======================================================================
# 4) Training / evaluation / tracked predictions
# ======================================================================
@torch.no_grad()
def evaluate(model, dl, criterion):
    model.eval()
    losses, preds, ys = [], [], []
    for b in dl:
        logits, _ = model(b["x"], b["orient"], b["mask"])
        losses.append(criterion(logits, b["y"]).item())
        preds.append(logits.argmax(1))
        ys.append(b["y"])
    preds, ys = torch.cat(preds).numpy(), torch.cat(ys).numpy()
    f1 = f1_score(ys, preds, average="macro", labels=list(range(N_CLASSES)), zero_division=0)
    return float(np.mean(losses)), f1


@torch.no_grad()
def predict_with_tracking(model, dl, meta):
    """Returns
    pred_df : one row per patient  (patient_id, label, pred, p0..p5)
    attn_df : one row per slice    (patient_id, label, orientation, slice_idx, slice_pos, attention)
    """
    model.eval()
    pred_rows, attn_parts = [], []
    for b in dl:
        logits, w = model(b["x"], b["orient"], b["mask"])
        probs = torch.softmax(logits, dim=1)
        for i, pid in enumerate(b["patient_id"]):
            row = {"patient_id": pid, "label": b["y"][i].item(), "pred": probs[i].argmax().item()}
            row.update({f"p{c}": probs[i, c].item() for c in range(probs.shape[1])})
            pred_rows.append(row)

            valid = b["mask"][i]
            rows = b["rows"][i][valid].numpy()            # row numbers -> metadata lookup
            a = meta.loc[rows, ["patient_id", "label", "orientation", "slice_idx", "slice_pos"]].copy()
            a["attention"] = w[i][valid].numpy()
            attn_parts.append(a)
    return pd.DataFrame(pred_rows), pd.concat(attn_parts, ignore_index=True)


def train_fold(Xs, meta, tr_ids, va_ids, n_orient, seed, epochs=60, lr=3e-4,
               weight_decay=1e-2, slice_drop=0.3, batch_size=8):
    torch.manual_seed(seed)
    train_ds = BagDataset(Xs, meta, tr_ids)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_bags)
    val_dl = DataLoader(BagDataset(Xs, meta, va_ids), batch_size=batch_size,
                        collate_fn=collate_bags)
    model = MultiOrientationMIL(input_dim=Xs.shape[1], n_orient=n_orient)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # class weights from patient counts: all 1.0 if balanced, compensates if you use more patients
    counts = np.bincount([it["label"] for it in train_ds.items], minlength=N_CLASSES).astype(float)
    class_w = torch.tensor(counts.sum() / (N_CLASSES * np.maximum(counts, 1)), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    history = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for b in train_dl:
            mask = drop_slices(b["mask"], slice_drop)
            optimizer.zero_grad()
            logits, _ = model(b["x"], b["orient"], mask)
            loss = criterion(logits, b["y"])
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        val_loss, val_f1 = evaluate(model, val_dl, criterion)
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)),
                        "val_loss": val_loss, "val_f1": val_f1})
    return model, pd.DataFrame(history)


def run_cv(X_all, meta, n_splits=5, seed=0, epochs=60, **train_kw):
    """Patient-level stratified CV on the merged data.
    The final epoch's model is used for the tracked predictions (no epoch picking on the
    validation fold). Look at `hist_df` to choose a sensible `epochs` value."""
    n_orient = meta["orient_id"].nunique()
    patients = meta.drop_duplicates("patient_id")[["patient_id", "label"]].reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    preds, attns, hists = [], [], []
    for fold, (tr, va) in enumerate(skf.split(patients, patients["label"])):
        tr_ids = patients["patient_id"].iloc[tr].to_numpy()
        va_ids = patients["patient_id"].iloc[va].to_numpy()

        # scaler is fitted on the training patients only
        tr_rows = meta.index[meta["patient_id"].isin(set(tr_ids))]
        scaler = StandardScaler().fit(X_all[tr_rows])
        Xs = scaler.transform(X_all).astype(np.float32)

        model, hist = train_fold(Xs, meta, tr_ids, va_ids, n_orient,
                                 seed=seed * 100 + fold, epochs=epochs, **train_kw)
        val_dl = DataLoader(BagDataset(Xs, meta, va_ids), batch_size=8, collate_fn=collate_bags)
        pred_df, attn_df = predict_with_tracking(model, val_dl, meta)
        for df in (pred_df, attn_df, hist):
            df["fold"], df["seed"] = fold, seed
        preds.append(pred_df); attns.append(attn_df); hists.append(hist)
        print(f"fold {fold}: val_f1(last epoch) = {hist['val_f1'].iloc[-1]:.3f}")

    return (pd.concat(preds, ignore_index=True),
            pd.concat(attns, ignore_index=True),
            pd.concat(hists, ignore_index=True))


# ======================================================================
# 5) Analysis helpers (all based on the tracked tables)
# ======================================================================
def attention_by_orientation(attn_df):
    """Share of attention each orientation receives per diagnosis (rows sum to 1).
    With equal slice counts per orientation, 1/3 each means 'no preference'."""
    per_patient = (attn_df.groupby(["patient_id", "label", "orientation"])["attention"]
                          .sum().unstack("orientation"))
    return per_patient.groupby("label").mean()


def top_slices(attn_df, k=3):
    """The k most attended slices per patient (orientation + slice index)."""
    return (attn_df.sort_values("attention", ascending=False)
                   .groupby("patient_id").head(k)
                   .sort_values(["patient_id", "attention"], ascending=[True, False]))
