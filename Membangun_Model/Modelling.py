import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)

DAGSHUB_USERNAME = "buildwithzee"
DAGSHUB_REPO     = "Eksperimen_SML_Zean-Ananda-Pratama"

dagshub.init(repo_owner=DAGSHUB_USERNAME,
             repo_name=DAGSHUB_REPO,
             mlflow=True)

mlflow.set_experiment("crop_recommendation_baseline")

# LOAD DATA
DATA_PATH = "crop_recommendation_preprocessing.csv"
df = pd.read_csv(DATA_PATH)

FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET   = 'label_encoded'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size : {X_train.shape}")
print(f"Test size  : {X_test.shape}")

def save_confusion_matrix(y_true, y_pred, labels, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.5)
    ax.set_title('Confusion Matrix', fontsize=14)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    return filename


def save_feature_importance(model, feature_names, filename="feature_importance.png"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(feature_names)),
           importances[indices],
           color='steelblue', edgecolor='white')
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=30, ha='right')
    ax.set_title('Feature Importance (Random Forest)', fontsize=13)
    ax.set_ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    return filename


def save_classification_report_json(y_true, y_pred, filename="classification_report.json"):
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(filename, 'w') as f:
        json.dump(report, f, indent=4)
    return filename


with mlflow.start_run(run_name="RandomForest_Baseline"):

    #parameter model
    params = {
        "n_estimators" : 100,
        "max_depth"    : None,
        "random_state" : 42,
        "n_jobs"       : -1
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #metrik utama
    acc       = accuracy_score(y_test, y_pred)
    f1_macro  = f1_score(y_test, y_pred, average='macro')
    f1_weight = f1_score(y_test, y_pred, average='weighted')
    prec      = precision_score(y_test, y_pred, average='weighted')
    rec       = recall_score(y_test, y_pred, average='weighted')

    #log parameter
    mlflow.log_params(params)

    #log metrik utama
    mlflow.log_metric("accuracy",           acc)
    mlflow.log_metric("f1_macro",           f1_macro)
    mlflow.log_metric("f1_weighted",        f1_weight)
    mlflow.log_metric("precision_weighted", prec)
    mlflow.log_metric("recall_weighted",    rec)

    #log artefak 1: Confusion Matrix
    label_names = [str(c) for c in sorted(y.unique())]
    cm_path = save_confusion_matrix(y_test, y_pred, label_names)
    mlflow.log_artifact(cm_path)

    #log artefak 2: Feature Importance
    fi_path = save_feature_importance(model, FEATURES)
    mlflow.log_artifact(fi_path)

    #log artefak 3: Classification Report JSON
    cr_path = save_classification_report_json(y_test, y_pred)
    mlflow.log_artifact(cr_path)

    #log model
    mlflow.sklearn.log_model(model, artifact_path="random_forest_model")

    print(f"Accuracy          : {acc:.4f}")
    print(f"F1 Macro          : {f1_macro:.4f}")
    print(f"F1 Weighted       : {f1_weight:.4f}")
    print(f"Precision Weighted: {prec:.4f}")
    print(f"Recall Weighted   : {rec:.4f}")
    print("Run selesai! Cek dashboard DagsHub.")