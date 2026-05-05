import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import entropy


# ============================================================
# Step 1: Load Dataset
# ============================================================

DATA_PATH = "creditcard.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ============================================================
# Step 2: Train Best Baseline Model
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest baseline model...")
model.fit(X_train, y_train)


# ============================================================
# Step 3: Drift Simulation Function
# ============================================================

def simulate_data_drift(X_batch, drift_strength=0.5):
    """
    Simulates artificial data drift by shifting selected feature distributions.
    Higher drift_strength = stronger drift.
    """

    X_drifted = X_batch.copy()

    drift_features = ["V1", "V2", "V3", "V4", "V10", "V11", "V12", "V14", "Amount"]

    for feature in drift_features:
        if feature in X_drifted.columns:
            noise = np.random.normal(
                loc=drift_strength,
                scale=drift_strength,
                size=len(X_drifted)
            )
            X_drifted[feature] = X_drifted[feature] + noise

    return X_drifted


# ============================================================
# Step 4: KL Divergence Drift Function
# ============================================================

def calculate_kl_divergence(reference, current, bins=50):
    """
    Calculates average KL divergence across all numerical features.
    """

    kl_values = []

    for col in reference.columns:
        ref_hist, bin_edges = np.histogram(reference[col], bins=bins, density=True)
        cur_hist, _ = np.histogram(current[col], bins=bin_edges, density=True)

        ref_hist = ref_hist + 1e-10
        cur_hist = cur_hist + 1e-10

        kl = entropy(ref_hist, cur_hist)
        kl_values.append(kl)

    return np.mean(kl_values)


# ============================================================
# Step 5: Monitoring Function
# ============================================================

def evaluate_batch(model, X_batch, y_batch, batch_id, drift_score):
    y_pred = model.predict(X_batch)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_batch)[:, 1]
        roc_auc = roc_auc_score(y_batch, y_prob)
    else:
        roc_auc = np.nan

    accuracy = accuracy_score(y_batch, y_pred)
    precision = precision_score(y_batch, y_pred, zero_division=0)
    recall = recall_score(y_batch, y_pred, zero_division=0)
    f1 = f1_score(y_batch, y_pred, zero_division=0)

    return {
        "Batch": batch_id,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "KL-Drift-Score": drift_score
    }


# ============================================================
# Step 6: Create Streaming Batches
# ============================================================

batch_size = 5000

X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

batches = []

for start in range(0, len(X_test_reset), batch_size):
    end = start + batch_size
    X_batch = X_test_reset.iloc[start:end].copy()
    y_batch = y_test_reset.iloc[start:end].copy()

    if len(X_batch) > 0:
        batches.append((X_batch, y_batch))

print("\nTotal Batches Created:", len(batches))


# ============================================================
# Step 7: Monitor Normal + Drifted Batches
# ============================================================

results = []

reference_data = X_train.copy()

for i, (X_batch, y_batch) in enumerate(batches, start=1):

    if i <= 5:
        drift_strength = 0.0
        X_current = X_batch.copy()
    elif i <= 10:
        drift_strength = 0.3
        X_current = simulate_data_drift(X_batch, drift_strength)
    else:
        drift_strength = 0.7
        X_current = simulate_data_drift(X_batch, drift_strength)

    drift_score = calculate_kl_divergence(reference_data, X_current)

    result = evaluate_batch(
        model=model,
        X_batch=X_current,
        y_batch=y_batch,
        batch_id=i,
        drift_score=drift_score
    )

    result["Drift_Strength"] = drift_strength

    results.append(result)

    print(
        f"Batch {i:02d} | "
        f"Drift Strength: {drift_strength} | "
        f"KL Score: {drift_score:.4f} | "
        f"F1: {result['F1-Score']:.4f} | "
        f"Recall: {result['Recall']:.4f} | "
        f"Precision: {result['Precision']:.4f}"
    )


# ============================================================
# Step 8: Save Monitoring Results
# ============================================================

results_df = pd.DataFrame(results)
results_df.to_csv("step2_drift_monitoring_results.csv", index=False)

print("\nStep 2 Results Saved: step2_drift_monitoring_results.csv")
print("\nFinal Step 2 Monitoring Results:")
print(results_df)