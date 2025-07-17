import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
import numpy as np
import joblib # For saving the model and scaler
import matplotlib.pyplot as plt # For plotting (optional, but good for verification)

# --- Configuration ---
PROCESSED_DATA_PATH = 'processed_behavior_features.csv'
MODEL_PATH = 'isolation_forest_model.joblib'
SCALER_PATH = 'standard_scaler.joblib' # Path to save the StandardScaler

# Load the processed DataFrame
try:
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Loaded '{PROCESSED_DATA_PATH}'.")
except FileNotFoundError:
    print(f"Error: '{PROCESSED_DATA_PATH}' not found.")
    print("Please ensure the file is in the same directory or run the feature engineering script first.")
    exit()

# Separate features (X) and labels (y)
X = df.drop('label', axis=1)
y = df['label']

# --- Identify numerical columns for scaling ---
# This logic is robust and will identify all numerical columns that are not binary (0/1)
# ensuring new numerical features like 'distance_from_trusted_loc' are included.
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

print("\n--- Starting Model Training (Isolation Forest) ---")

# --- Model Selection & Training: Isolation Forest ---
contamination_rate = y_train.value_counts(normalize=True).get(1, 0.01) # Get the proportion of anomalous samples, default to 0.01 if no anomalies
print(f"Contamination rate for training: {contamination_rate:.4f}")

model = IsolationForest(contamination=contamination_rate, random_state=42, n_estimators=100)
model.fit(X_train)

# --- Prediction ---
y_test_pred = model.predict(X_test)
y_test_pred_mapped = np.where(y_test_pred == -1, 1, 0) # Map -1/1 to 0/1 for comparison

print("\n--- Model Evaluation ---")
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_test_pred_mapped, target_names=['Normal (0)', 'Anomalous (1)']))

print("\nConfusion Matrix on Test Set:")
print(confusion_matrix(y_test, y_test_pred_mapped))

# ROC AUC Score
y_test_scores = model.decision_function(X_test) # Get raw anomaly scores
roc_auc = roc_auc_score(y_test, -y_test_scores) # Use negative scores for AUC as higher means more confident in anomaly
print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")

# --- Saving the Model and Scaler ---
print("\n--- Saving the Model and Scaler ---")

# Save the trained model
joblib.dump(model, MODEL_PATH)
print(f"Model successfully saved to '{MODEL_PATH}'")

# Save the fitted scaler
joblib.dump(scaler, SCALER_PATH)
print(f"Scaler successfully saved to '{SCALER_PATH}'")

print("\nModel training, evaluation, and saving complete. The model and scaler are now ready for deployment!")
