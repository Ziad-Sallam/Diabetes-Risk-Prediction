import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix -- {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    fname = name.replace(' ', '_').replace('--', '-') + '_CM.png'
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"  Saved -> {fname}")


def tune_softmax(X_train, y_train, X_val, y_val, tag=''):
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]

    best_score, best_C = -1, None

    print(f"\n[Softmax{tag}] Tuning hyperparameters on validation set ...")
    print(f"{'C':>8}  {'Val F1-Macro':>12}")
    print('-' * 24)

    for C in C_values:
        clf = LogisticRegression(
            C=C,
            solver='lbfgs',
            max_iter=1000,
            random_state=RANDOM_SEED
        )
        clf.fit(X_train, y_train)
        score = f1_score(y_val, clf.predict(X_val), average='macro')
        print(f"{C:>8.3f}  {score:>12.4f}")

        if score > best_score:
            best_score = score
            best_C     = C

    print(f"\n  Best config: C={best_C}, Val F1-Macro={best_score:.4f}")
    return best_C

if __name__ == '__main__':


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

    # Baseline Softmax 
    print('\n' + '-'*58)
    print('  SOFTMAX REGRESSION -- BASELINE')
    print('-'*58)

    best_C_base = tune_softmax(X_tr_base, y_tr_base, X_val, y_val, tag=' Baseline')
    softmax_base = LogisticRegression(
        C=best_C_base,
        solver='lbfgs',
        max_iter=1000,
        random_state=RANDOM_SEED
    )
    softmax_base.fit(X_tr_base, y_tr_base)
    res_base = evaluate(softmax_base, X_te, y_te, 'Softmax - Baseline')
    plot_cm(y_te, res_base['y_pred'], 'Softmax - Baseline')

    # SMOTE Softmax
    print('\n' + '-'*58)
    print('  SOFTMAX REGRESSION -- SMOTE')
    print('-'*58)

    best_C_smote = tune_softmax(X_tr_smote, y_tr_smote, X_val, y_val, tag=' SMOTE')
    softmax_smote = LogisticRegression(
        C=best_C_smote,
        solver='lbfgs',
        max_iter=1000,
        random_state=RANDOM_SEED
    )
    softmax_smote.fit(X_tr_smote, y_tr_smote)
    res_smote = evaluate(softmax_smote, X_te, y_te, 'Softmax - SMOTE')
    plot_cm(y_te, res_smote['y_pred'], 'Softmax - SMOTE')

    print('\n' + '-'*70)
    print('  SOFTMAX RESULTS SUMMARY')
    print('-'*70)
    summary = pd.DataFrame([
        {k: round(v, 4) if isinstance(v, float) else v
         for k, v in r.items() if k != 'y_pred'}
        for r in [res_base, res_smote]
    ])
    print(summary.to_string(index=False))
    summary.to_csv('softmax_results_summary.csv', index=False)
    print('\n  Saved -> softmax_results_summary.csv')