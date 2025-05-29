import os
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- Semilla ---
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Configuramos las rutas ---
wsi_dir = "slidesM"
df = pd.read_csv(os.path.join(wsi_dir, 'completo.csv'), sep=";")
df = df.rename(columns={"filename": "slide_id", "clase": "label"})

# --- Codificamos etiquetas ---
df['label'] = df['label'].astype(str)
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])

# --- Cargar características ---
dir_h5 = 'outputTotal/40x_512px_0px_overlap/features_resnet50'

def extract_mean_features(row):
    slide_id = row["slide_id"].replace(".svs", "")
    path = os.path.join(dir_h5, slide_id + ".h5")
    try:
        with h5py.File(path, "r") as f:
            feats = f["features"][:]
            return np.mean(feats, axis=0)
    except Exception as e:
        print(f"Error cargando {path}: {e}")
        return None

print("Extrayendo características para todos los datos...")
df['features'] = df.apply(extract_mean_features, axis=1)
df = df[df['features'].notnull()].reset_index(drop=True)
X = np.stack(df['features'].values)
y = df['label_enc'].values

# --- Dataset personalizado ---
class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]  # (1, 1024), label

class SimpleBinaryClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = x.squeeze(1)
        return self.net(x).squeeze(1)

# --- Cross-validation ---
print("Iniciando validación cruzada 5-Fold...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

all_metrics = []
fold = 1

for train_idx, test_idx in skf.split(X, y):
    print(f"\n--- Fold {fold} ---")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    smote = SMOTE(random_state=SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    train_dataset = FeatureDataset(X_train_res, y_train_res)
    test_dataset = FeatureDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleBinaryClassifier().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    all_outputs, all_labels = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            predicted = (outputs > 0).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    pred_labels = (all_outputs > 0).astype(int)

    auc = roc_auc_score(all_labels, all_outputs)
    acc = correct / total
    bal_acc = balanced_accuracy_score(all_labels, pred_labels)
    f1 = f1_score(all_labels, pred_labels)

    print(f"Fold {fold} - AUC: {auc:.4f}, Accuracy: {acc:.4f}, Balanced Accuracy: {bal_acc:.4f}, F1-score: {f1:.4f}")
    cm = confusion_matrix(all_labels, pred_labels)
    print("\nConfusion Matrix:")
    print(cm)
    all_metrics.append((auc, acc, bal_acc, f1))

    fold += 1

# --- Resultados finales ---
aucs, accs, bals, f1s = zip(*all_metrics)
print("\n--- Resultados Promediados de Validación Cruzada ---")
print(f"AUC Promedio: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")
print(f"Accuracy Promedio: {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")
print(f"Balanced Accuracy Promedio: {np.mean(bals):.4f} (+/- {np.std(bals):.4f})")
print(f"F1-score Promedio: {np.mean(f1s):.4f} (+/- {np.std(f1s):.4f})")
