import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

RANDOM_SEED = 42

# This class handles all data preparation steps: loading, splitting, scaling, feature selection, imbalance handling, and saving.
#*** Usage:
# x = DataPrep(
#     file_path='dataset/diabetes_012_health_indicators_BRFSS2015.csv',
#     feature_selection_method='manual',  # Options: 'manual', 'automatic', or None
#     imbalance_method='SMOTE',           # Options: 'oversampling', 'undersampling', 'SMOTE', or None
#     k=15,                              # Only used if feature_selection_method='automatic' 
#     save_dir='dataset/preprocessedData'                  # Directory to save preprocessed data
# )

class DataPrep:
    def __init__(self, file_path, feature_selection_method=None, imbalance_method=None, k=15, save_dir='dataset'):
        self.df = pd.read_csv(file_path)
        
        self.feature_names = self.df.drop("Diabetes_012", axis=1).columns.tolist()
        
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.split_data()
        
        self.X_train_scaled, self.X_val_scaled, self.X_test_scaled = self.feature_scaling()

        if feature_selection_method == 'manual':
            self.X_train_scaled, self.X_val_scaled, self.X_test_scaled = self.feature_selection_manual()
        elif feature_selection_method == 'automatic':
            self.X_train_scaled, self.X_val_scaled, self.X_test_scaled = self.feature_selection_automatic(k)

        if imbalance_method == 'oversampling':
            self.X_train_scaled, self.y_train = self.feature_imbalance_oversampling()
        elif imbalance_method == 'undersampling':
            self.X_train_scaled, self.y_train = self.feature_imbalance_undersampling()
        elif imbalance_method == 'SMOTE':
            self.X_train_scaled, self.y_train = self.feature_imbalance_SMOTE()

        # 5. Save the data
        self.save_preprocessed_data(save_dir)
        print(f"Data preparation complete! Files saved in '{save_dir}/'")

    def split_data(self):
        # Split the data into features and target variable
        X = self.df.drop("Diabetes_012", axis=1)
        y = self.df["Diabetes_012"]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(2/3), random_state=RANDOM_SEED, stratify=y_temp)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def feature_scaling(self):
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(self.X_train)
        X_val_scaled = scaler.transform(self.X_val)
        X_test_scaled = scaler.transform(self.X_test)

        return X_train_scaled, X_val_scaled, X_test_scaled

    def feature_selection_manual(self):
        features_to_drop = ['Healthcare', 'Sex', 'NoDocbcCost', 'Fruits', 'PhysHlth', 'DiffWalk', 'Education']
        indices_to_drop = [self.feature_names.index(f) for f in features_to_drop if f in self.feature_names]
        
        # Update self.feature_names for saving later
        self.feature_names = [f for i, f in enumerate(self.feature_names) if i not in indices_to_drop]

        X_train_sel = np.delete(self.X_train_scaled, indices_to_drop, axis=1)
        X_val_sel = np.delete(self.X_val_scaled, indices_to_drop, axis=1)
        X_test_sel = np.delete(self.X_test_scaled, indices_to_drop, axis=1)

        return X_train_sel, X_val_sel, X_test_sel
    
    def feature_selection_automatic(self, k=15):
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(self.X_train_scaled, self.y_train)
        X_val_selected = selector.transform(self.X_val_scaled)
        X_test_selected = selector.transform(self.X_test_scaled)

        mask = selector.get_support()
        self.feature_names = [self.feature_names[i] for i, selected in enumerate(mask) if selected]

        return X_train_selected, X_val_selected, X_test_selected

    def feature_imbalance_oversampling(self):
        ros = RandomOverSampler(random_state=RANDOM_SEED)
        X_resampled, y_resampled = ros.fit_resample(self.X_train_scaled, self.y_train)
        return X_resampled, y_resampled

    def feature_imbalance_undersampling(self):
        rus = RandomUnderSampler(random_state=RANDOM_SEED)
        X_resampled, y_resampled = rus.fit_resample(self.X_train_scaled, self.y_train)
        return X_resampled, y_resampled
    
    def feature_imbalance_SMOTE(self):
        smote = SMOTE(random_state=RANDOM_SEED)
        X_resampled, y_resampled = smote.fit_resample(self.X_train_scaled, self.y_train)
        return X_resampled, y_resampled
    
    def save_preprocessed_data(self, save_dir):
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        df_train = pd.DataFrame(self.X_train_scaled, columns=self.feature_names)
        df_train['Diabetes_012'] = self.y_train.values

        df_val = pd.DataFrame(self.X_val_scaled, columns=self.feature_names)
        df_val['Diabetes_012'] = self.y_val.values

        df_test = pd.DataFrame(self.X_test_scaled, columns=self.feature_names)
        df_test['Diabetes_012'] = self.y_test.values

        # Save the preprocessed data to new CSV files
        df_train.to_csv(os.path.join(save_dir, 'preprocessed_train.csv'), index=False)
        df_val.to_csv(os.path.join(save_dir, 'preprocessed_val.csv'), index=False)
        df_test.to_csv(os.path.join(save_dir, 'preprocessed_test.csv'), index=False)