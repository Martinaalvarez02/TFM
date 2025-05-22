import os
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler 
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

# --- Vemos distribución de clases antes del oversampling ---
print("\nDistribución de clases en el conjunto de entrenamiento antes del oversampling:")
print(pd.Series(y_train).value_counts())

# --- Aplicamos Random OverSampling ---
print("Aplicando Random OverSampling para balancear clases...")
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

# --- Vemos distribución de clases después del oversampling ---
print("\nDistribución de clases en el conjunto de entrenamiento después del oversampling:")
print(pd.Series(y_train_res).value_counts())

# --- Parámetros para GridSearch ---
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 10],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'n_estimators': [100, 200, 300]
}

# --- Configuramos XGBoost y GridSearchCV ---
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42
)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3, verbose=2, n_jobs=-1)

# --- Entrenamos modelo con GridSearch ---
grid_search.fit(X_train_res, y_train_res)

# --- Mejor combinación de parámetros ---
print("\nMejores parámetros encontrados:")
print(grid_search.best_params_)

# --- Mejor modelo entrenado ---
best_model = grid_search.best_estimator_

# --- Predicciones ---
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
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

# --- Matriz de confusión ---
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Clase 0', 'Clase 1'], yticklabels=['Clase 0', 'Clase 1'])
plt.xlabel('Etiqueta predicha')
plt.ylabel('Etiqueta real')
plt.title('Matriz de Confusión')
plt.tight_layout()
plt.show()