import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from trident.slide_encoder_models import ABMILSlideEncoder

# --- Configuración de rutas ---
wsi_dir = "slidesM"
coords_dir = "outputTotal"

# --- Cargar CSV y renombrar columnas ---
df = pd.read_csv(os.path.join(wsi_dir, 'completo.csv'), sep=";")
df = df.rename(columns={"filename": "slide_id", "clase": "label"})

# División
train, test = train_test_split(df, test_size=0.25, random_state=42, stratify=df['label'])

# Semillas
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Modelo base ---
class BaseModel(nn.Module):
    def __init__(self, input_feature_dim=1024, n_heads=1, head_dim=512, dropout=0., gated=True, hidden_dim=256):
        super().__init__()
        self.feature_encoder = ABMILSlideEncoder(
            input_feature_dim=input_feature_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
            gated=gated
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        features = self.feature_encoder(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features).squeeze(1)
        return logits

# Dataset
class H5Dataset(Dataset):
    def __init__(self, feats_path, df, split, num_features=512):
        self.df = df
        self.feats_path = feats_path
        self.num_features = num_features
        self.split = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = row['slide_id'].replace('.svs', '')
        file_path = os.path.join(self.feats_path, slide_id + '.h5')

        try:
            with h5py.File(file_path, "r") as f:
                features = torch.from_numpy(f["features"][:])
        except (OSError, KeyError):
            features = torch.zeros((self.num_features, 1024))
            label = torch.tensor(0., dtype=torch.float32)
            return features, label

        if self.split == 'train':
            num_available = features.shape[0]
            indices = (torch.randperm(num_available, generator=torch.Generator().manual_seed(SEED))[:self.num_features]
                        if num_available >= self.num_features else
                        torch.randint(num_available, (self.num_features,), generator=torch.Generator().manual_seed(SEED)))
            features = features[indices]

        label = torch.tensor(row["label"], dtype=torch.float32)
        return features, label

# --- Función de entrenamiento ---
def train_model(model, train_loader, num_epochs, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.
        for features, labels in train_loader:
            features, labels = {'features': features.to(device)}, labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# --- Evaluación del modelo ---
def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels, preds = [], []

    with torch.no_grad():
        for features, labels in test_loader:
            features = {'features': features.to(device)}
            output = model(features)
            predicted = (output > 0).cpu().numpy()  # Convertir a 0 o 1
            preds.extend(predicted)
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    print("\n--- Evaluación Final ---")
    print(classification_report(all_labels, preds))
    cm = confusion_matrix(all_labels, preds)
    print("Matriz de confusión:\n", cm)

    # Visualización
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Clase 0', 'Clase 1'], yticklabels=['Clase 0', 'Clase 1'])
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.show()

    return preds, all_labels

# --- MAIN ---
def main():
    feats_path = os.path.join(coords_dir, '40x_512px_0px_overlap/features_resnet50')
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # División del entrenamiento por clase
    train_0 = train[train['label'] == 0].copy()
    train_1 = train[train['label'] == 1].copy()

    train_loader_0 = DataLoader(H5Dataset(feats_path, train_0, "train"), batch_size=batch_size, shuffle=True)
    train_loader_1 = DataLoader(H5Dataset(feats_path, train_1, "train"), batch_size=batch_size, shuffle=True)
    test_loader_0 = DataLoader(H5Dataset(feats_path, test[test['label'] == 0], "test"), batch_size=1, shuffle=False)
    test_loader_1 = DataLoader(H5Dataset(feats_path, test[test['label'] == 1], "test"), batch_size=1, shuffle=False)

    # Crear y entrenar el modelo para la clase 0
    print("\n--- Entrenando modelo para clase 0 ---")
    model0 = BaseModel()
    train_model(model0, train_loader_0, num_epochs=30, device=device)

    # Evaluación del modelo para clase 0
    print("\n--- Evaluando modelo para clase 0 ---")
    preds_0, labels_0 = evaluate_model(model0, test_loader_0, device)

    # Crear y entrenar el modelo para la clase 1
    print("\n--- Entrenando modelo para clase 1 ---")
    model1 = BaseModel()
    train_model(model1, train_loader_1, num_epochs=30, device=device)

    # Evaluación del modelo para clase 1
    print("\n--- Evaluando modelo para clase 1 ---")
    preds_1, labels_1 = evaluate_model(model1, test_loader_1, device)

    # Concatenar las predicciones de ambos modelos
    all_preds = np.concatenate([preds_0, preds_1])
    all_labels = np.concatenate([labels_0, labels_1])

    print("\n--- Evaluación Combinada del Sistema ---")
    print(classification_report(all_labels, all_preds))
    cm_combined = confusion_matrix(all_labels, all_preds)
    print("Matriz de confusión combinada:\n", cm_combined)

    # Visualización de la matriz de confusión combinada
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_combined, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Clase 0', 'Clase 1'], yticklabels=['Clase 0', 'Clase 1'])
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión Combinada')
    plt.tight_layout()
    plt.show()

    # Entrenamiento con ambos conjuntos de clases
    print("\n--- Entrenando el modelo con ambas clases ---")
    train_combined = train.copy()
    train_loader_combined = DataLoader(H5Dataset(feats_path, train_combined, "train"), batch_size=batch_size, shuffle=True)

    # Crear y entrenar el modelo con ambas clases
    model_combined = BaseModel()
    train_model(model_combined, train_loader_combined, num_epochs=30, device=device)

    # Evaluación del modelo con el conjunto de prueba combinado
    print("\n--- Evaluando modelo combinado con los datos de prueba ---")
    test_loader_combined = DataLoader(H5Dataset(feats_path, test, "test"), batch_size=1, shuffle=False)
    preds_combined, labels_combined = evaluate_model(model_combined, test_loader_combined, device)

    # Evaluación final
    print("\n--- Evaluación Final del Modelo Combinado ---")
    print(classification_report(labels_combined, preds_combined))
    cm_final = confusion_matrix(labels_combined, preds_combined)
    print("Matriz de confusión final:\n", cm_final)

    # Visualización de la matriz de confusión final
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Clase 0', 'Clase 1'], yticklabels=['Clase 0', 'Clase 1'])
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión Final')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()