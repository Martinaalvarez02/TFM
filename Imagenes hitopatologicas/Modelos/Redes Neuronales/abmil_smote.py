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

from trident.slide_encoder_models import ABMILSlideEncoder
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

# --- Dividimos en train/test ---
train_df, test_df = train_test_split(df, test_size=0.25, stratify=df['label'], random_state=SEED)

# --- Codificamos etiquetas ---
le = LabelEncoder()
train_df['label_enc'] = le.fit_transform(train_df['label'].astype(str))
test_df['label_enc'] = le.transform(test_df['label'].astype(str))

# --- Dataset personalizado para ABMIL ---
class H5Dataset(Dataset):
    def __init__(self, feats_path, df, num_features=512):
        self.df = df
        self.feats_path = feats_path
        self.num_features = num_features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = row['slide_id'].replace('.svs', '')
        file_path = os.path.join(self.feats_path, slide_id + '.h5')

        try:
            with h5py.File(file_path, "r") as f:
                features = torch.from_numpy(f["features"][:])
        except (OSError, KeyError) as e:
            print(f"Error cargando {file_path}: {e}")
            features = torch.zeros((self.num_features, 1024))
            label = torch.tensor(0., dtype=torch.float32)
            return features, label

        num_available = features.shape[0]
        if num_available >= self.num_features:
            indices = torch.randperm(num_available, generator=torch.Generator().manual_seed(SEED))[:self.num_features]
        else:
            indices = torch.randint(num_available, (self.num_features,), generator=torch.Generator().manual_seed(SEED))
        features = features[indices]

        label = torch.tensor(row["label_enc"], dtype=torch.float32)
        return features, label

# --- Modelo ABMIL ---
class BinaryClassificationModel(nn.Module):
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

    def forward(self, x, return_raw_attention=False):
        if return_raw_attention:
            features, attn = self.feature_encoder(x, return_raw_attention=True)
        else:
            features = self.feature_encoder(x)        
        features = features.view(features.size(0), -1)  
        logits = self.classifier(features).squeeze(1)
        return (logits, attn) if return_raw_attention else logits

# --- Función para extraer embeddings por imagen ---
def extract_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for features, label in dataloader:
            features = features.to(device)
            emb = model.feature_encoder({'features': features}) 
            emb = emb.view(emb.size(0), -1)
            embeddings.append(emb.cpu().numpy())
            labels.append(label.numpy())
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    return embeddings, labels


# --- Entrenamiento y evaluación ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8

    train_dataset = H5Dataset(dir_h5, train_df)
    test_dataset = H5Dataset(dir_h5, test_df)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = BinaryClassificationModel(input_feature_dim=1024).to(device)

    # --- Extraemos embeddings ---
    print("Extrayendo características de entrenamiento...")
    X_train, y_train = extract_embeddings(model, train_loader, device)

    print("Extrayendo características de prueba...")
    X_test, y_test = extract_embeddings(model, test_loader, device)

    # --- Aplicamos SMOTE ---
    print("Aplicando SMOTE para balancear clases...")
    smote = SMOTE(random_state=SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    

    # --- Dataset para clasificación ---
    class FeatureDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = FeatureDataset(X_train_res, y_train_res)
    test_dataset = FeatureDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- Clasificador ---
    classifier = nn.Sequential(
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    ).to(device)


    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.05)
    num_epochs = 50

    # --- Entrenamiento ---
    for epoch in range(num_epochs):
        classifier.train()
        total_loss = 0.
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(features).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # --- Evaluación ---
    classifier.eval()
    all_outputs, all_labels = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = classifier(features).squeeze(1)
            predicted = (outputs > 0).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_outputs)
    accuracy = correct / total

    print(f"\nTest AUC: {auc:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")


    # Reporte
    pred_labels = (all_outputs > 0).astype(int)
    balanced_acc = balanced_accuracy_score(all_labels, pred_labels)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, pred_labels, target_names=le.classes_))

    cm = confusion_matrix(all_labels, pred_labels)
    print("\nConfusion Matrix:")
    print(cm)

    # --- Visualización ---
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