import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the processed DataFrame
try:
    df = pd.read_csv('processed_behavior_features.csv')
    print("Loaded 'processed_behavior_features.csv'.")
except FileNotFoundError:
    print("Error: 'processed_behavior_features.csv' not found.")
    print("Please ensure the file is in the same directory or provide the correct path.")
    exit()

# Separate features (X) and labels (y)
X = df.drop('label', axis=1)
y = df['label']

# --- Identify numerical columns for scaling ---
# Get all numerical columns that are not one-hot encoded (which are already 0 or 1)
# One-hot encoded columns (e.g., browser_Chrome, os_Windows) are already on a 0/1 scale,
# and don't typically need standard scaling.
numerical_cols_to_scale = [col for col in X.columns if X[col].dtype in ['float64', 'int64'] and not (X[col].isin([0, 1]).all() and X[col].nunique() <= 2)]

print(f"\nNumerical columns to scale: {numerical_cols_to_scale}")

# --- Feature Scaling ---
scaler = StandardScaler()
X_scaled = X.copy() # Create a copy to avoid modifying original X directly

# Apply scaling only to the identified numerical columns
X_scaled[numerical_cols_to_scale] = scaler.fit_transform(X[numerical_cols_to_scale])

print("\n--- Feature Scaling Complete ---")
print("Scaled features head (only showing scaled columns for clarity):")
print(X_scaled[numerical_cols_to_scale].head())

# --- Train-Test Split ---
# We'll use a stratification strategy to ensure both train and test sets have similar proportions
# of normal (0) and anomalous (1) samples. This is important for imbalanced datasets.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- Train-Test Split Complete ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

print("\nTraining label distribution (y_train):")
print(y_train.value_counts(normalize=True))

print("\nTesting label distribution (y_test):")
print(y_test.value_counts(normalize=True))

# At this point, X_train, X_test, y_train, y_test are ready for model training.
print("\nYour data is now preprocessed and split into training and testing sets!")
print("You can proceed to model training using X_train and y_train, and then evaluate with X_test and y_test.")