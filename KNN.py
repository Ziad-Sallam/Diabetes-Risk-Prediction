import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report
)

import DataPrep

RANDOM_SEED  = 42
CLASS_NAMES  = ['No Diabetes', 'Prediabetes', 'Diabetes']
DATASET_PATH = 'dataset/diabetes_012_health_indicators_BRFSS2015.csv'


def evaluate(model, X_test, y_test, name):
    """Print metrics and return a result dict."""
    y_pred = model.predict(X_test)

    acc         = accuracy_score(y_test, y_pred)
    f1_micro    = f1_score(y_test, y_pred, average='micro')
    f1_macro    = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    print(f"\n{'='*58}")
    print(f"  {name}")
    print(f"{'='*58}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  F1-Micro        : {f1_micro:.4f}")
    print(f"  F1-Macro        : {f1_macro:.4f}")
    print(f"  F1-Weighted     : {f1_weighted:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    return dict(name=name, accuracy=acc, f1_micro=f1_micro,
                f1_macro=f1_macro, f1_weighted=f1_weighted, y_pred=y_pred)


def plot_cm(y_test, y_pred, name):
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix -- {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    fname = name.replace(' ', '_').replace('--', '-') + '_CM.png'
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"  Saved -> {fname}")


def tune_knn(X_train, y_train, X_val, y_val, tag=''):
    """
    Grid-search over k values and distance metrics.
    Selection criterion: highest Macro-F1 on the validation set.
    """
    k_values = [3, 5, 11, 21]
    metrics  = ['euclidean', 'manhattan']

    best_score, best_params = -1, {}

    print(f"\n[KNN{tag}] Tuning hyperparameters on validation set ...")
    print(f"{'k':>4}  {'metric':<12}  {'Val F1-Macro':>12}")
    print('-' * 32)

    for k in k_values:
        for metric in metrics:
            clf = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)
            clf.fit(X_train, y_train)
            score = f1_score(y_val, clf.predict(X_val), average='macro')
            print(f"{k:>4}  {metric:<12}  {score:>12.4f}")

            if score > best_score:
                best_score  = score
                best_params = {'k': k, 'metric': metric}

    print(f"\n  Best config: k={best_params['k']}, "
          f"metric={best_params['metric']}, Val F1-Macro={best_score:.4f}")
    return best_params


if __name__ == '__main__':

    # Load data 
    print('Loading BASELINE data (no imbalance handling) ...')
    base = DataPrep.DataPrep(
        file_path=DATASET_PATH,
        feature_selection_method='automatic',
        imbalance_method=None,
        k=15,
        save_dir='dataset/preprocessedData_baseline'
    )

    print('\nLoading SMOTE data ...')
    smote = DataPrep.DataPrep(
        file_path=DATASET_PATH,
        feature_selection_method='automatic',
        imbalance_method='SMOTE',
        k=15,
        save_dir='dataset/preprocessedData_smote'
    )

    X_tr_base,  X_val, X_te = base.X_train_scaled,  base.X_val_scaled,  base.X_test_scaled
    y_tr_base,  y_val, y_te = base.y_train,          base.y_val,          base.y_test

    X_tr_smote, y_tr_smote  = smote.X_train_scaled, smote.y_train

    # Baseline KNN 
    print('\n' + '-'*58)
    print('  KNN -- BASELINE')
    print('-'*58)

    params_base = tune_knn(X_tr_base, y_tr_base, X_val, y_val, tag=' Baseline')
    knn_base = KNeighborsClassifier(
        n_neighbors=params_base['k'],
        metric=params_base['metric'],
        n_jobs=-1
    )
    knn_base.fit(X_tr_base, y_tr_base)
    res_base = evaluate(knn_base, X_te, y_te, 'KNN - Baseline')
    plot_cm(y_te, res_base['y_pred'], 'KNN - Baseline')

    # SMOTE KNN 
    print('\n' + '-'*58)
    print('  KNN -- SMOTE')
    print('-'*58)

    params_smote = tune_knn(X_tr_smote, y_tr_smote, X_val, y_val, tag=' SMOTE')
    knn_smote = KNeighborsClassifier(
        n_neighbors=params_smote['k'],
        metric=params_smote['metric'],
        n_jobs=-1
    )
    knn_smote.fit(X_tr_smote, y_tr_smote)
    res_smote = evaluate(knn_smote, X_te, y_te, 'KNN - SMOTE')
    plot_cm(y_te, res_smote['y_pred'], 'KNN - SMOTE')

    print('\n' + '-'*70)
    print('  KNN RESULTS SUMMARY')
    print('-'*70)
    summary = pd.DataFrame([
        {k: round(v, 4) if isinstance(v, float) else v
         for k, v in r.items() if k != 'y_pred'}
        for r in [res_base, res_smote]
    ])
    print(summary.to_string(index=False))
    summary.to_csv('knn_results_summary.csv', index=False)
    print('\n  Saved -> knn_results_summary.csv')
