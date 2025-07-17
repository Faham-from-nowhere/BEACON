import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Load the processed DataFrame (assuming it was saved from the previous step)
try:
    df = pd.read_csv('processed_behavior_features.csv')
    print("Loaded 'processed_behavior_features.csv'.")
except FileNotFoundError:
    print("Error: 'processed_behavior_features.csv' not found.")
    print("Please ensure the file is in the same directory or run the feature engineering script first.")
    exit()

# Separate features (X) and labels (y)
X = df.drop('label', axis=1)
y = df['label']

# Identify numerical columns for scaling
numerical_cols_to_scale = [col for col in X.columns if X[col].dtype in ['float64', 'int64'] and not (X[col].isin([0, 1]).all() and X[col].nunique() <= 2)]

# Apply scaling only to the identified numerical columns
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numerical_cols_to_scale] = scaler.fit_transform(X[numerical_cols_to_scale])

# Train-Test Split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- Starting Model Training (Isolation Forest) ---")

# --- Model Selection & Training: Isolation Forest ---
# Isolation Forest works by isolating anomalies. It assigns a 'decision_function' score,
# where lower scores indicate more anomalous points.
# 'contamination' is a hyperparameter representing the expected proportion of outliers in the data.
# It's usually set based on prior knowledge or can be tuned.
# For our synthetic data, we know the anomalous samples make up 200/1200 = 0.1666...
# We'll set contamination based on the training data's anomaly ratio.

contamination_rate = y_train.value_counts(normalize=True)[1] # Get the proportion of anomalous samples in y_train

print(f"Contamination rate for training: {contamination_rate:.4f}")

# Initialize and train the Isolation Forest model
# random_state for reproducibility
model = IsolationForest(contamination=contamination_rate, random_state=42, n_estimators=100)
model.fit(X_train)

# --- Prediction ---
# decision_function gives the anomaly score: lower means more anomalous
y_train_scores = model.decision_function(X_train)
y_test_scores = model.decision_function(X_test)

# predict returns -1 for outliers and 1 for inliers based on the contamination threshold
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Convert Isolation Forest's -1/1 predictions to our 0/1 label format (1 for anomalous, 0 for normal)
# Isolation Forest: -1 = anomaly, 1 = normal
# Our labels: 1 = anomaly, 0 = normal
y_train_pred_mapped = np.where(y_train_pred == -1, 1, 0)
y_test_pred_mapped = np.where(y_test_pred == -1, 1, 0)


print("\n--- Model Evaluation ---")

# --- Evaluation Metrics ---
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_test_pred_mapped, target_names=['Normal (0)', 'Anomalous (1)']))

print("\nConfusion Matrix on Test Set:")
print(confusion_matrix(y_test, y_test_pred_mapped))

# ROC AUC Score
# For ROC AUC, we need probabilities or decision scores.
# decision_function gives the raw anomaly scores; these are often used directly for AUC.
roc_auc = roc_auc_score(y_test, -y_test_scores) # Negative scores mean more anomalous
print(f"\nROC AUC Score on Test Set: {roc_auc:.4f}")

# --- Plotting ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, -y_test_scores) # Use negative scores as higher means more confident in anomaly
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_curve.png') # Save the plot
plt.close() # Close the plot to prevent it from displaying in certain environments

# --- Plotting Precision-Recall Curve ---
precision, recall, _ = precision_recall_curve(y_test, -y_test_scores)
pr_auc = auc(recall, precision)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True)
plt.savefig('precision_recall_curve.png') # Save the plot
plt.close() # Close the plot

print("\nEvaluation plots (roc_curve.png, precision_recall_curve.png) saved.")

print("\nModel training and initial evaluation complete.")
print("You now have a trained Isolation Forest model!")
print("The next step could be hyperparameter tuning, deeper analysis of false positives/negatives, or deploying the model.")