import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from trident.slide_encoder_models import ABMILSlideEncoder
from sklearn.metrics import balanced_accuracy_score


# --- Configuramos las rutas ---
wsi_dir = "slidesM"
coords_dir = "outputTotal"

# --- Cargamos CSV y renombramos columnas ---
df = pd.read_csv(os.path.join(wsi_dir, 'completo.csv'), sep=";")
df = df.rename(columns={"filename": "slide_id", "clase": "label"})

# --- Semillas ---
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Modelo ---
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

# --- Dataset ---
class H5Dataset(Dataset):
    def __init__(self, feats_path, df, split, num_features=512):
        self.df = df.reset_index(drop=True)
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
        except (OSError, KeyError) as e:
            print(f"Error cargando {file_path}: {e}")
            features = torch.zeros((self.num_features, 1024))
            label = torch.tensor(0., dtype=torch.float32)
            return features, label

        if self.split == 'train':
            num_available = features.shape[0]
            if num_available >= self.num_features:
                indices = torch.randperm(num_available, generator=torch.Generator().manual_seed(SEED))[:self.num_features]
            else:
                indices = torch.randint(num_available, (self.num_features,), generator=torch.Generator().manual_seed(SEED))
            features = features[indices]

        label = torch.tensor(row["label"], dtype=torch.float32)
        return features, label

# --- Entrenamiento y Validación cruzada ---
dir_h5 = 'outputTotal/40x_512px_0px_overlap'

def main():
    feats_path = os.path.join(dir_h5, 'features_resnet50')
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

    all_aucs = []
    all_accs = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df, df['label'])):
        print(f"\n--- Fold {fold+1} ---")

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        train_loader = DataLoader(H5Dataset(feats_path, train_df, "train"), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(H5Dataset(feats_path, test_df, "test"), batch_size=1, shuffle=False)

        model = BinaryClassificationModel(input_feature_dim=1024).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.05)

        num_epochs = 30
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

        # Evaluación
        model.eval()
        all_labels, all_outputs = [], []
        correct, total = 0, 0

        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = {'features': features.to(device)}, labels.to(device)
                outputs = model(features)
                predicted = (outputs > 0).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_outputs = np.concatenate(all_outputs)
        all_labels = np.concatenate(all_labels)
        predicted_labels = (all_outputs > 0).astype(int) 

        auc = roc_auc_score(all_labels, all_outputs)
        accuracy = correct / total
        bal_acc = balanced_accuracy_score(all_labels, predicted_labels)  

        all_aucs.append(auc)
        all_accs.append(accuracy)

        print(f"\nFold {fold+1} - AUC: {auc:.4f}, Accuracy: {accuracy:.4f} - Balanced Accuracy: {bal_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, predicted_labels))


        cm = confusion_matrix(all_labels, predicted_labels)
        print("Confusion Matrix:")
        print(cm)

        # Matriz de confusión
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['LIHC', 'CHOL'], yticklabels=['LIHC', 'CHOL'])
        plt.title(f'Matriz de Confusión - Fold {fold+1}')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.tight_layout()
        plt.show()

    print("\n--- Validación cruzada completada ---")
    print(f"Media AUC: {np.mean(all_aucs):.4f} ± {np.std(all_aucs):.4f}")
    print(f"Media Accuracy: {np.mean(all_accs):.4f} ± {np.std(all_accs):.4f}")

if __name__ == "__main__":
    main()