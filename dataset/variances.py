import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# 1. Load Data
df = pd.read_csv('dataset/diabetes_012_health_indicators_BRFSS2015.csv')

# 2. Calculate Raw Variances
raw_vars = df.var().sort_values(ascending=False)

# 3. Calculate Normalized Variances (Scaling features to 0-1 range first)
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
normalized_vars = df_scaled.var().sort_values(ascending=False)

# 4. Plotting
plt.figure(figsize=(12, 10))

# Plot Raw
plt.subplot(2, 1, 1)
sns.barplot(
    x=raw_vars.values, 
    y=raw_vars.index, 
    hue=raw_vars.index,  # Assign y variable to hue
    palette='magma', 
    legend=False         # Disable legend to keep it clean
)
plt.title('Raw Feature Variances (Scale Dependent)')
plt.xscale('log') # Log scale because 'Area' is huge compared to others

# Plot Normalized
plt.subplot(2, 1, 2)
sns.barplot(
    x=normalized_vars.values, 
    y=normalized_vars.index, 
    hue=normalized_vars.index, # Assign y variable to hue
    palette='viridis', 
    legend=False               # Disable legend
)
plt.title('Normalized Variances (After Min-Max Scaling)')

plt.tight_layout()
plt.savefig('variances_comparison.png')

from sklearn.feature_selection import VarianceThreshold

# 0.01 threshold means the variance is less than 1% 
# (Common for binary data where one class is 99% of the total)
selector = VarianceThreshold(threshold=0.01)
X_reduced = selector.fit_transform(df_scaled)

print(f"Features kept: {X_reduced.shape[1]} out of {df.shape[1]}")