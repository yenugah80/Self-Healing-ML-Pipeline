import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ============================================================
# Step 1: Load Dataset
# ============================================================

DATA_PATH = "creditcard.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\nClass Distribution:")
print(df["Class"].value_counts())

print("\nFraud Percentage:")
print(df["Class"].value_counts(normalize=True) * 100)


# ============================================================
# Step 2: Prepare Features and Target
# ============================================================

X = df.drop("Class", axis=1)
y = df["Class"]

# Scale Amount and Time
scaler = StandardScaler()
X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])


# ============================================================
# Step 3: Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# ============================================================
# Step 4: Define Models
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
        eval_metric="logloss",
        random_state=42
    )
}


# ============================================================
# Step 5: Evaluation Function
# ============================================================

def evaluate_model(model_name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = np.nan

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)

    print("\n====================================================")
    print(f"Model: {model_name}")
    print("====================================================")
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    }


# ============================================================
# Step 6: Train and Evaluate Models
# ============================================================

results = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)

    result = evaluate_model(model_name, model, X_test, y_test)
    results.append(result)


# ============================================================
# Step 7: Save Results
# ============================================================

results_df = pd.DataFrame(results)
results_df.to_csv("step1_baseline_results.csv", index=False)

print("\nBaseline Results Saved: step1_baseline_results.csv")
print("\nFinal Results:")
print(results_df)