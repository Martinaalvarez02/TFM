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
from sklearn.metrics import balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from trident.slide_encoder_models import ABMILSlideEncoder

# --- Configuración de rutas y semillas ---
wsi_dir = "slidesM"
coords_dir = "outputTotal"
dir_h5 = 'outputTotal/40x_512px_0px_overlap'
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Cargamos CSVs ---
df = pd.read_csv(os.path.join(wsi_dir, 'completo.csv'), sep=";")
df = df.rename(columns={"filename": "slide_id", "clase": "label"})

genomic_df = pd.read_csv(os.path.join(wsi_dir, 'expresion_genes_4.csv'), index_col=0)
genomic_df.index = genomic_df.index.str.extract(r'(^[a-f0-9\-]+)', expand=False)

# --- Dataset multimodal ---
class H5GenomicDataset(Dataset):
    def __init__(self, feats_path, df, split, genomic_data, num_features=512):
        self.df = df
        self.feats_path = feats_path
        self.genomic_data = genomic_data
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
        except:
            features = torch.zeros((self.num_features, 1024))
            gene_vec = torch.zeros(4)
            label = torch.tensor(0., dtype=torch.float32)
            return features, gene_vec, label

        if self.split == 'train':
            num_available = features.shape[0]
            if num_available >= self.num_features:
                indices = torch.randperm(num_available)[:self.num_features]
            else:
                indices = torch.randint(num_available, (self.num_features,))
            features = features[indices]

        if slide_id in self.genomic_data.index:
            gene_row = self.genomic_data.loc[slide_id]
            gene_vec = torch.tensor(gene_row[['APLN', 'LINP1', 'LHX6', 'AC022211.2']].values.astype(np.float32))
        else:
            gene_vec = torch.zeros(4)

        label = torch.tensor(row["label"], dtype=torch.float32)
        return features, gene_vec, label

# --- Modelo multimodal ---
class BinaryMultiModalModel(nn.Module):
    def __init__(self, input_feature_dim=1024, genomic_dim=4, hidden_dim=256):
        super().__init__()
        self.feature_encoder = ABMILSlideEncoder(
            input_feature_dim=input_feature_dim,
            n_heads=1,
            head_dim=512,
            dropout=0.0,
            gated=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_feature_dim + genomic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_img, x_gene):
        img_features = self.feature_encoder(x_img)
        img_features = img_features.view(img_features.size(0), -1)
        combined = torch.cat((img_features, x_gene), dim=1)
        logits = self.classifier(combined).squeeze(1)
        return logits

# --- Entrenamiento y evaluación ---
def main():
    feats_path = os.path.join(dir_h5, 'features_resnet50')
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, shuffle=True, stratify=df['label'])

    train_loader = DataLoader(H5GenomicDataset(feats_path, train_df, "train", genomic_df),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(H5GenomicDataset(feats_path, test_df, "test", genomic_df),
                             batch_size=1, shuffle=False)

    model = BinaryMultiModalModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.
        for features, genes, labels in train_loader:
            features = {'features': features.to(device)}
            genes = genes.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features, genes)
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
        for features, genes, labels in test_loader:
            features = {'features': features.to(device)}
            genes = genes.to(device)
            labels = labels.to(device)
            outputs = model(features, genes)
            predicted = (outputs > 0).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_outputs)
    accuracy = correct / total
    print(f"Test AUC: {auc:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    predicted_labels = (all_outputs > 0).astype(int)
    balanced_acc = balanced_accuracy_score(all_labels, predicted_labels)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, predicted_labels))

    cm = confusion_matrix(all_labels, predicted_labels)
    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['LIHC', 'CHOL'], yticklabels=['LIHC', 'CHOL'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()