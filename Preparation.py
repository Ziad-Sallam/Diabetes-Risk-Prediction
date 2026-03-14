import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42

df = pd.read_csv("dataset/diabetes_012_health_indicators_BRFSS2015.csv")

# Split the data into features and target variable
X = df.drop("Diabetes_012", axis=1)
y = df["Diabetes_012"]

#split the data into training, validation and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(2/3), random_state=RANDOM_SEED)

#feature scaling
scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

