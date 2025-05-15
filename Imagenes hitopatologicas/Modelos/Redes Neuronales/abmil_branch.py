
# import pandas as pd
# import numpy as np
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import h5py
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# from trident.slide_encoder_models import ABMILSlideEncoder

# # --- Configuración de rutas ---
# wsi_dir = "slidesM"
# coords_dir = "outputTotal"

# # --- Cargar CSV y renombrar columnas ---
# df = pd.read_csv(os.path.join(wsi_dir, 'completo.csv'), sep=";")
# df = df.rename(columns={"filename": "slide_id", "clase": "label"})

# print(df.head())
# print(df.columns)

# # --- Dividir en entrenamiento y prueba ---
# #train, test = train_test_split(df, test_size=0.25, random_state=42, shuffle=True)
# train, test = train_test_split(df, test_size=0.25, random_state=42, shuffle=True, stratify=df['label'])
# print("TRAIN:\n", train.head(), "\n")
# print("TEST:\n", test.head(), "\n")

# # --- Semillas ---
# SEED = 1234
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# # --- Modelo ---
# import torch.nn.functional as F

# class BinaryClassificationModel(nn.Module):
#     def __init__(self, input_feature_dim=1024, n_heads=1, head_dim=512, dropout=0.25, gated=True, hidden_dim=256):
#         super().__init__()
#         self.feature_encoder = ABMILSlideEncoder(
#             input_feature_dim=input_feature_dim,
#             n_heads=n_heads,
#             head_dim=head_dim,
#             dropout=dropout,
#             gated=gated
#         )
#         self.branch1 = nn.Sequential(
#             nn.Linear(input_feature_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
#         self.branch2 = nn.Sequential(
#             nn.Linear(input_feature_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
#         self.final_classifier = nn.Linear(hidden_dim * 2, 1)

#     def forward(self, x, return_raw_attention=False):
#         if return_raw_attention:
#             features, attn = self.feature_encoder(x, return_raw_attention=True)
#         else:
#             features = self.feature_encoder(x)

#         features = features.view(features.size(0), -1)
#         features = F.normalize(features, dim=1) 

#         out1 = self.branch1(features)
#         out2 = self.branch2(features)
#         combined = torch.cat((out1, out2), dim=1)
#         logits = self.final_classifier(combined).squeeze(1)
#         return (logits, attn) if return_raw_attention else logits



# # --- Dataset ---
# class H5Dataset(Dataset):
#     def __init__(self, feats_path, df, split, num_features=512):
#         self.df = df
#         self.feats_path = feats_path
#         self.num_features = num_features
#         self.split = split

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         slide_id = row['slide_id'].replace('.svs', '')
#         file_path = os.path.join(self.feats_path, slide_id + '.h5')

#         try:
#             with h5py.File(file_path, "r") as f:
#                 features = torch.from_numpy(f["features"][:])
#         except (OSError, KeyError) as e:
#             print(f"Error cargando {file_path}: {e}")
#             features = torch.zeros((self.num_features, 1024))  
#             label = torch.tensor(0., dtype=torch.float32)
#             return features, label

#         if self.split == 'train':
#             num_available = features.shape[0]
#             if num_available >= self.num_features:
#                 indices = torch.randperm(num_available, generator=torch.Generator().manual_seed(SEED))[:self.num_features]
#             else:
#                 indices = torch.randint(num_available, (self.num_features,), generator=torch.Generator().manual_seed(SEED))
#             features = features[indices]

#         label = torch.tensor(row["label"], dtype=torch.float32)
#         return features, label


# # --- Entrenamiento ---
# dir_h5 = 'outputTotal/40x_512px_0px_overlap'

# def main():
#     feats_path = os.path.join(dir_h5, 'features_resnet50')
#     batch_size = 8
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     train_loader = DataLoader(H5Dataset(feats_path, train, "train"), batch_size=batch_size, shuffle=True, worker_init_fn=lambda _: np.random.seed(SEED))
#     test_loader = DataLoader(H5Dataset(feats_path, test, "test"), batch_size=1, shuffle=False, worker_init_fn=lambda _: np.random.seed(SEED))

#     model = BinaryClassificationModel(input_feature_dim=1024).to(device)
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.005)  # Puedes ajustar este valor
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

#     num_epochs = 50
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0.
#         for features, labels in train_loader:
#             features, labels = {'features': features.to(device)}, labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(features)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
        
#         avg_loss = total_loss / len(train_loader)
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
#         # Actualizar el scheduler
#         scheduler.step(avg_loss)  

#     model.eval()
#     all_labels, all_outputs = [], []
#     correct, total = 0, 0

#     with torch.no_grad():
#         for features, labels in test_loader:
#             features, labels = {'features': features.to(device)}, labels.to(device)
#             outputs = model(features)
#             probs = torch.sigmoid(outputs)
#             predicted = (probs > 0.5).float()
#             correct += (predicted == labels).sum().item()
#             total += labels.size(0)
#             all_outputs.append(probs.cpu().numpy())
#             all_labels.append(labels.cpu().numpy())

#     all_outputs = np.concatenate(all_outputs)
#     all_labels = np.concatenate(all_labels)
#     auc = roc_auc_score(all_labels, all_outputs)
#     accuracy = correct / total

#     print(f"Test AUC: {auc:.4f}")
#     print(f"Test Accuracy: {accuracy:.4f}")

#     # Reporte de clasificación
#     predicted_labels = (all_outputs > 0.5).astype(int)
#     print("\nClassification Report:")
#     print(classification_report(all_labels, predicted_labels))

#     # Matriz de confusión
#     cm = confusion_matrix(all_labels, predicted_labels)
#     print("\nConfusion Matrix:")
#     print(cm)
    


# if __name__ == "__main__":
#     main()


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
from sklearn.metrics import balanced_accuracy_score

# --- Configuración de rutas ---
wsi_dir = "slidesM"
coords_dir = "outputTotal"

# --- Cargar CSV y renombrar columnas ---
df = pd.read_csv(os.path.join(wsi_dir, 'completo.csv'), sep=";")
df = df.rename(columns={"filename": "slide_id", "clase": "label"})

print(df.head())
print(df.columns)

# --- Dividir en entrenamiento y prueba ---
train, test = train_test_split(df, test_size=0.25, random_state=42, shuffle=True, stratify=df['label'])
print("TRAIN:\n", train.head(), "\n")
print("TEST:\n", test.head(), "\n")

# --- Semillas ---
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Modelo ---
import torch.nn.functional as F

class DualPerceptronModel(nn.Module):
    def __init__(self, input_feature_dim=1024, n_heads=1, head_dim=512, dropout=0.25, gated=True, hidden_dim=256):
        super().__init__()
        self.feature_encoder = ABMILSlideEncoder(
            input_feature_dim=input_feature_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
            gated=gated
        )
        # Perceptrón para clase 0
        self.perceptron0 = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        # Perceptrón para clase 1
        self.perceptron1 = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, return_raw_attention=False):
        if return_raw_attention:
            features, attn = self.feature_encoder(x, return_raw_attention=True)
        else:
            features = self.feature_encoder(x)

        features = features.view(features.size(0), -1)
        features = F.normalize(features, dim=1)

        out0 = self.perceptron0(features)
        out1 = self.perceptron1(features)

        # Resta: si out1 > out0, predice clase 1; si out0 > out1, predice clase 0
        logits = out1 - out0
        logits = logits.squeeze(1)

        return (logits, attn) if return_raw_attention else logits




# --- Dataset ---
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


# --- Entrenamiento ---
dir_h5 = 'outputTotal/40x_512px_0px_overlap'

def main():
    feats_path = os.path.join(dir_h5, 'features_resnet50')
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(H5Dataset(feats_path, train, "train"), batch_size=batch_size, shuffle=True, worker_init_fn=lambda _: np.random.seed(SEED))
    test_loader = DataLoader(H5Dataset(feats_path, test, "test"), batch_size=1, shuffle=False, worker_init_fn=lambda _: np.random.seed(SEED))

    model = DualPerceptronModel(input_feature_dim=1024).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    num_epochs = 50
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
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Actualizar el scheduler
        scheduler.step(avg_loss)  #


    model.eval()
    all_labels, all_outputs = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = {'features': features.to(device)}, labels.to(device)
            outputs = model(features)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_outputs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_outputs)
    accuracy = correct / total

    print(f"Test AUC: {auc:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Reporte de clasificación
    predicted_labels = (all_outputs > 0.5).astype(int)
    balanced_acc = balanced_accuracy_score(all_labels, predicted_labels)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, predicted_labels))

    # Matriz de confusión
    cm = confusion_matrix(all_labels, predicted_labels)
    print("\nConfusion Matrix:")
    print(cm)
    


if __name__ == "__main__":
    main()
