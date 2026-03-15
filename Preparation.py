import DataPrep

x = DataPrep.DataPrep(
    file_path='dataset/diabetes_012_health_indicators_BRFSS2015.csv',
    feature_selection_method='automatic',  # Options: 'manual', 'automatic', or None
    imbalance_method='SMOTE',           # Options: 'oversampling', 'undersampling', 'SMOTE', or None
    k=15,                              # Only used if feature_selection_method='automatic'
    save_dir='dataset/preprocessedData'                  # Directory to save preprocessed data
)
