import os
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score

# --- Semilla ---
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Configuramos las rutas ---
wsi_dir = "slidesM"
coords_dir = "outputTotal"
dir_h5 = 'outputTotal/40x_512px_0px_overlap/features_resnet50'

# --- Cargamos CSV ---
df = pd.read_csv(os.path.join(wsi_dir, 'completo.csv'), sep=";")
df = df.rename(columns={"filename": "slide_id", "clase": "label"})

# --- Cargamos los datos de expresión génica ---
genomic_df = pd.read_csv(os.path.join(wsi_dir, 'expresion_genes_4.csv'), index_col=0)
genomic_df.index = genomic_df.index.str.extract(r'(^[a-f0-9\-]+)', expand=False)

# --- Dividimos en train/test ---
train_df, test_df = train_test_split(df, test_size=0.25, stratify=df['label'], random_state=SEED)

# --- Extraemos características promedio (mean pooling) y datos genómicos ---
def extract_mean_features(df_split, genomic_df):
    X_img, X_gene, y, slide_ids = [], [], [], []
    for _, row in df_split.iterrows():
        slide_id = row["slide_id"].replace(".svs", "")
        path = os.path.join(dir_h5, slide_id + ".h5")
        try:
            with h5py.File(path, "r") as f:
                feats = f["features"][:]
                feats_mean = np.mean(feats, axis=0)
        except Exception as e:
            print(f"Error cargando {path}: {e}")
            continue

        # Genomic vector
        if slide_id in genomic_df.index:
            gene_vec = genomic_df.loc[slide_id][['APLN', 'LINP1', 'LHX6', 'AC022211.2']].values.astype(np.float32)
        else:
            gene_vec = np.zeros(4, dtype=np.float32)

        X_img.append(feats_mean)
        X_gene.append(gene_vec)
        y.append(str(row["label"]))
        slide_ids.append(slide_id)
    
    X_img = np.array(X_img)
    X_gene = np.array(X_gene)
    X_combined = np.concatenate([X_img, X_gene], axis=1)
    return X_combined, np.array(y), slide_ids

# --- Extraemos características ---
print("Extrayendo características de entrenamiento...")
X_train, y_train, slide_ids_train = extract_mean_features(train_df, genomic_df)

print("Extrayendo características de prueba...")
X_test, y_test, slide_ids_test = extract_mean_features(test_df, genomic_df)

# --- Codificamos etiquetas ---
le = LabelEncoder()
y_train = y_train.astype(str)
y_test = y_test.astype(str)
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# --- Aplicamos SMOTE para balancear clases ---
print("Aplicando SMOTE para balancear clases...")
smote = SMOTE(random_state=SEED)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train_enc)

class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]  # (1, 1028), label

class SimpleBinaryClassifier(nn.Module):
    def __init__(self, input_dim=1028, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = x.squeeze(1)
        return self.net(x).squeeze(1)

# --- Entrenamiento ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleBinaryClassifier().to(device)

    train_dataset = FeatureDataset(X_train_res, y_train_res)
    test_dataset = FeatureDataset(X_test, y_test_enc)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)  
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # --- Evaluación ---
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
    auc = roc_auc_score(all_labels, all_outputs)
    accuracy = correct / total

    ## Resultados
    print(f"\nTest AUC: {auc:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    pred_labels = (all_outputs > 0).astype(int)
    balanced_acc = balanced_accuracy_score(all_labels, pred_labels)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, pred_labels, target_names=le.classes_))

    cm = confusion_matrix(all_labels, pred_labels)
    print("\nConfusion Matrix:")
    print(cm)

    # --- Visualización (si se pudiera)---
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Etiqueta predicha')
    plt.ylabel('Etiqueta real')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()