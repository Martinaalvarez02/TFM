import os
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score


df = pd.read_csv("slidesM/completo.csv", sep=";")
df = df.rename(columns={"filename": "slide_id", "clase": "label"})

train, test = train_test_split(df, test_size=0.25, stratify=df["label"], random_state=42)

features_path = "outputTotal/40x_512px_0px_overlap/features_resnet50"

def extract_features(df_split):
    X, y = [], []
    for _, row in df_split.iterrows():
        slide_id = row["slide_id"].replace(".svs", "")
        path = os.path.join(features_path, slide_id + ".h5")
        try:
            with h5py.File(path, "r") as f:
                feats = f["features"][:]
                feats_mean = np.mean(feats, axis=0)  
                X.append(feats_mean)
                y.append(row["label"])
        except Exception as e:
            print(f"Error cargando {path}: {e}")
    return np.array(X), np.array(y)

# --- Extraemos características ---
print("Extrayendo características de entrenamiento...")
X_train, y_train = extract_features(train)
print("Extrayendo características de prueba...")
X_test, y_test = extract_features(test)

# --- Entrenamos modelo XGBoost ---
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    colsample_bytree = 0.8,
    min_child_weight=1,
    subsample=0.8
    
)

model.fit(X_train, y_train)

# --- Predicciones ---
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

# --- Evaluación ---
auc = roc_auc_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)

print(f"\nAUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)