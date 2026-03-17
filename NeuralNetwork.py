import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
import os
import json

# ---------------------------
# Data Preparation
# ---------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LR = 0.0005
EPOCHS = 40
imbalance_methods = ['oversampling', 'undersampling', 'SMOTE', None]
feature_selection_methods = [ 'automatic']
ks = [15, 18, 20, 21]

# ---------------------------
# Neural Network Architecture
# ---------------------------
class DiabetesNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.model(x)

for imbalance_method in imbalance_methods:
    for feature_selection_method in feature_selection_methods:
        for k in ks:
            MODEL_SAVE_PATH = f'models/{imbalance_method}_{feature_selection_method}_{k}/diabetes_nn_model_best_{imbalance_method}_{feature_selection_method}_{k}.pth'
            METRICS_SAVE_PATH = f'models/{imbalance_method}_{feature_selection_method}_{k}/diabetes_metrics_{imbalance_method}_{feature_selection_method}_{k}.json'
            print(f"Using device: {DEVICE}")
            os.makedirs(os.path.dirname(METRICS_SAVE_PATH), exist_ok=True)

            # ---------------------------
            # Load datasets
            # ---------------------------
            train_df = pd.read_csv(f"dataset/preprocessedData_{imbalance_method}_{feature_selection_method}_{k}/preprocessed_train.csv")
            val_df = pd.read_csv(f"dataset/preprocessedData_{imbalance_method}_{feature_selection_method}_{k}/preprocessed_val.csv")
            test_df = pd.read_csv(f"dataset/preprocessedData_{imbalance_method}_{feature_selection_method}_{k}/preprocessed_test.csv")

            X_train = train_df.drop("Diabetes_012", axis=1).values
            y_train = train_df["Diabetes_012"].values
            X_val = val_df.drop("Diabetes_012", axis=1).values
            y_val = val_df["Diabetes_012"].values
            X_test = test_df.drop("Diabetes_012", axis=1).values
            y_test = test_df["Diabetes_012"].values

            input_size = X_train.shape[1]

            # Convert to tensors
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.long)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.long)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.long)

            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)


            model = DiabetesNN(input_size).to(DEVICE)

            # ---------------------------
            # Class Weights for Imbalance
            # ---------------------------
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train.numpy()),
                y=y_train.numpy()
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            optimizer = optim.AdamW(model.parameters(), lr=LR)

            # ---------------------------
            # Training Loop with Best Macro F1 Saving
            # ---------------------------
            best_macro_f1 = 0.0

            for epoch in range(EPOCHS):
                model.train()
                total_loss = 0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                # Validation
                model.eval()
                preds_val, true_val = [], []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(DEVICE)
                        outputs = model(X_batch)
                        predicted = torch.argmax(outputs, dim=1)
                        preds_val.extend(predicted.cpu().numpy())
                        true_val.extend(y_batch.numpy())

                val_acc = accuracy_score(true_val, preds_val)
                val_macro_f1 = f1_score(true_val, preds_val, average="macro")
                print(f"Epoch {epoch+1}/{EPOCHS} | Loss {total_loss:.3f} | Val Acc {val_acc:.4f} | Val Macro F1 {val_macro_f1:.4f}")

                # Save best model
                if val_macro_f1 > best_macro_f1:
                    best_macro_f1 = val_macro_f1
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    print(f"Saved new best model at epoch {epoch+1} with Macro F1 {best_macro_f1:.4f}")

            # ---------------------------
            # Test Evaluation
            # ---------------------------
            model.load_state_dict(torch.load(MODEL_SAVE_PATH))
            model.eval()
            with torch.no_grad():
                outputs = model(X_test.to(DEVICE))
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

            accuracy = accuracy_score(y_test, preds)
            micro_f1 = f1_score(y_test, preds, average="micro")
            macro_f1 = f1_score(y_test, preds, average="macro")
            weighted_f1 = f1_score(y_test, preds, average="weighted")
            class_report = classification_report(y_test, preds, output_dict=True)
            conf_matrix = confusion_matrix(y_test, preds)

            print("\nTest Accuracy:", accuracy)
            print("Micro F1:", micro_f1)
            print("Macro F1:", macro_f1)
            print("Weighted F1:", weighted_f1)
            print("\nClassification Report:\n", classification_report(y_test, preds))
            print("\nConfusion Matrix:\n", conf_matrix)

            # ---------------------------
            # Save metrics
            # ---------------------------
            metrics = {
                "accuracy": accuracy,
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "classification_report": class_report,
                "confusion_matrix": conf_matrix.tolist()
            }

            with open(METRICS_SAVE_PATH, "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"Metrics saved to {METRICS_SAVE_PATH}")
            print(f"Best model saved to {MODEL_SAVE_PATH}")