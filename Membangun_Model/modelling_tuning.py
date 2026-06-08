import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score, StratifiedKFold)
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize

DAGSHUB_USERNAME = "buildwithzee"
DAGSHUB_REPO     = "Eksperimen_SML_Zean-Ananda-Pratama"

dagshub.init(repo_owner=DAGSHUB_USERNAME,
             repo_name=DAGSHUB_REPO,
             mlflow=True)

mlflow.set_experiment("crop_recommendation_tuning")

#LOAD DATA
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


def save_confusion_matrix(y_true, y_pred, labels, filename="confusion_matrix_tuning.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.5)
    ax.set_title('Confusion Matrix (Tuned Model)', fontsize=14)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    return filename


def save_feature_importance(model, feature_names, filename="feature_importance_tuning.png"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(feature_names)),
           importances[indices],
           color='mediumseagreen', edgecolor='white')
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=30, ha='right')
    ax.set_title('Feature Importance (Tuned Random Forest)', fontsize=13)
    ax.set_ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    return filename


def save_classification_report_json(y_true, y_pred, filename="classification_report_tuning.json"):
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(filename, 'w') as f:
        json.dump(report, f, indent=4)
    return filename


def save_cv_scores_plot(cv_scores, filename="cv_scores.png"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(cv_scores) + 1), cv_scores,
           color='coral', edgecolor='white')
    ax.axhline(cv_scores.mean(), color='navy', linestyle='--',
                label=f'Mean CV = {cv_scores.mean():.4f}')
    ax.set_title('Cross-Validation Accuracy per Fold', fontsize=13)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(range(1, len(cv_scores) + 1))
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    return filename


def save_gridsearch_heatmap(cv_results, param_grid, filename="gridsearch_heatmap.png"):
    results_df = pd.DataFrame(cv_results)

    pivot = results_df.pivot_table(
        index='param_max_depth',
        columns='param_n_estimators',
        values='mean_test_score'
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlGnBu',
                ax=ax, linewidths=0.5)
    ax.set_title('GridSearch CV Mean Score\n(n_estimators vs max_depth)', fontsize=13)
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('max_depth')
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    return filename


# HYPERPARAMETER TUNING (GridSearchCV)
print("Memulai GridSearchCV")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth'   : [5, 10, 20, None],
    'min_samples_split': [2, 5]
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator  = RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid = param_grid,
    cv         = cv_strategy,
    scoring    = 'accuracy',
    n_jobs     = -1,
    verbose    = 2,
    return_train_score=True
)

start_time = time.time()
grid_search.fit(X_train, y_train)
tuning_duration = time.time() - start_time

best_model  = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"\nBest params : {best_params}")
print(f"Tuning time : {tuning_duration:.2f} detik")

cv_scores = cross_val_score(best_model, X_train, y_train,
                             cv=cv_strategy, scoring='accuracy')


best_model.fit(X_train, y_train)
y_pred      = best_model.predict(X_test)
y_pred_prob = best_model.predict_proba(X_test)


acc       = accuracy_score(y_test, y_pred)
f1_macro  = f1_score(y_test, y_pred, average='macro')
f1_weight = f1_score(y_test, y_pred, average='weighted')
prec      = precision_score(y_test, y_pred, average='weighted')
rec       = recall_score(y_test, y_pred, average='weighted')
classes   = sorted(y.unique())
y_bin     = label_binarize(y_test, classes=classes)
roc_auc   = roc_auc_score(y_bin, y_pred_prob, multi_class='ovr', average='weighted')

#mlflow
with mlflow.start_run(run_name="RandomForest_Tuned"):

    #log parameter terbaik
    mlflow.log_params(best_params)
    mlflow.log_param("cv_folds",         5)
    mlflow.log_param("tuning_strategy",  "GridSearchCV")
    mlflow.log_param("tuning_duration_s", round(tuning_duration, 2))

    #log metrik utama (sama dengan autolog)
    mlflow.log_metric("accuracy",              acc)
    mlflow.log_metric("f1_macro",              f1_macro)
    mlflow.log_metric("f1_weighted",           f1_weight)
    mlflow.log_metric("precision_weighted",    prec)
    mlflow.log_metric("recall_weighted",       rec)

    #log metrik tambahan (melebihi autolog)
    mlflow.log_metric("roc_auc_weighted",      roc_auc)
    mlflow.log_metric("cv_mean_accuracy",      cv_scores.mean())
    mlflow.log_metric("cv_std_accuracy",       cv_scores.std())
    mlflow.log_metric("best_cv_score",         grid_search.best_score_)

    #artefak 1: Confusion Matrix
    label_names = [str(c) for c in classes]
    cm_path = save_confusion_matrix(y_test, y_pred, label_names)
    mlflow.log_artifact(cm_path)

    #artefak 2: Feature Importance
    fi_path = save_feature_importance(best_model, FEATURES)
    mlflow.log_artifact(fi_path)

    #artefak 3: Classification Report JSON
    cr_path = save_classification_report_json(y_test, y_pred)
    mlflow.log_artifact(cr_path)

    #artefak tambahan 4: CV Scores Plot
    cv_plot_path = save_cv_scores_plot(cv_scores)
    mlflow.log_artifact(cv_plot_path)

    #artefak tambahan 5: GridSearch Heatmap
    gs_path = save_gridsearch_heatmap(grid_search.cv_results_, param_grid)
    mlflow.log_artifact(gs_path)

    #log model
    mlflow.sklearn.log_model(best_model, artifact_path="random_forest_tuned")

    print(f"Accuracy          : {acc:.4f}")
    print(f"F1 Macro          : {f1_macro:.4f}")
    print(f"F1 Weighted       : {f1_weight:.4f}")
    print(f"Precision Weighted: {prec:.4f}")
    print(f"Recall Weighted   : {rec:.4f}")
    print(f"ROC-AUC Weighted  : {roc_auc:.4f}")
    print(f"CV Mean Accuracy  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("Tuning selesai! Cek dashboard DagsHub.")