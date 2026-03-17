import DataPrep

imbalance_methods = ['oversampling', 'undersampling', 'SMOTE', None]
feature_selection_methods = [ 'automatic']
ks = [15, 18, 20, 21]

for imbalance_method in imbalance_methods:
    for feature_selection_method in feature_selection_methods:
        for k in ks :
            x = DataPrep.DataPrep(
                file_path='dataset/diabetes_012_health_indicators_BRFSS2015.csv',
                feature_selection_method=feature_selection_method,  # Options: 'manual', 'automatic', or None
                imbalance_method=imbalance_method,           # Options: 'oversampling', 'undersampling', 'SMOTE', or None
                k=k,                              # Only used if feature_selection_method='automatic'
                save_dir=f'dataset/preprocessedData_{imbalance_method}_{feature_selection_method}_{k}'                  # Directory to save preprocessed data
            )