import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.stats import entropy


# ============================================================
# Config
# ============================================================

DATA_PATH = "creditcard.csv"

BATCH_SIZE = 5000

F1_THRESHOLD = 0.70
KL_THRESHOLD = 0.05
CRITICAL_KL_THRESHOLD = 0.10

RANDOM_STATE = 42


# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv(DATA_PATH)

X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

X_train, X_stream, y_train, y_stream = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

X_stream = X_stream.reset_index(drop=True)
y_stream = y_stream.reset_index(drop=True)


# ============================================================
# Drift Simulation
# ============================================================

def simulate_data_drift(X_batch, drift_strength=0.5):
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
# KL Divergence
# ============================================================

def calculate_kl_divergence(reference, current, bins=50):
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
# Evaluation
# ============================================================

def evaluate_model(model, X_batch, y_batch):
    y_pred = model.predict(X_batch)

    if hasattr(model, "predict_proba") and len(np.unique(y_batch)) > 1:
        y_prob = model.predict_proba(X_batch)[:, 1]
        roc_auc = roc_auc_score(y_batch, y_prob)
    else:
        roc_auc = np.nan

    return {
        "Accuracy": accuracy_score(y_batch, y_pred),
        "Precision": precision_score(y_batch, y_pred, zero_division=0),
        "Recall": recall_score(y_batch, y_pred, zero_division=0),
        "F1-Score": f1_score(y_batch, y_pred, zero_division=0),
        "ROC-AUC": roc_auc
    }


# ============================================================
# Agentic Decision Controller
# ============================================================

def agentic_decision_controller(f1, kl):
    if kl >= CRITICAL_KL_THRESHOLD:
        return "Manual Review", "Critical drift detected; human review recommended."

    elif f1 < F1_THRESHOLD and kl > KL_THRESHOLD:
        return "Retrain", "Performance degradation and significant drift detected."

    elif f1 < F1_THRESHOLD and kl <= KL_THRESHOLD:
        return "Tune Hyperparameters", "Performance degradation detected without significant drift."

    elif f1 >= F1_THRESHOLD and kl > KL_THRESHOLD:
        return "Warning", "Drift detected but performance remains acceptable."

    else:
        return "Continue", "Model performance and drift score are acceptable."


# ============================================================
# Candidate Models for Retraining
# ============================================================

def get_candidate_models(y_train):
    scale_pos_weight = y_train.value_counts()[0] / max(y_train.value_counts()[1], 1)

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        )
    }


# ============================================================
# Retraining Engine
# ============================================================

def retrain_and_select_best(X_train_updated, y_train_updated, X_validation, y_validation):
    candidate_models = get_candidate_models(y_train_updated)

    best_model = None
    best_model_name = None
    best_f1 = -1
    candidate_results = []

    for model_name, candidate_model in candidate_models.items():
        print(f"    Retraining candidate: {model_name}")

        candidate_model.fit(X_train_updated, y_train_updated)

        metrics = evaluate_model(candidate_model, X_validation, y_validation)
        candidate_f1 = metrics["F1-Score"]

        candidate_results.append({
            "Candidate_Model": model_name,
            "Validation_F1": candidate_f1,
            "Validation_Precision": metrics["Precision"],
            "Validation_Recall": metrics["Recall"],
            "Validation_ROC_AUC": metrics["ROC-AUC"]
        })

        if candidate_f1 > best_f1:
            best_f1 = candidate_f1
            best_model = candidate_model
            best_model_name = model_name

    return best_model, best_model_name, best_f1, candidate_results


# ============================================================
# Create Stream Batches
# ============================================================

batches = []

for start in range(0, len(X_stream), BATCH_SIZE):
    end = start + BATCH_SIZE
    X_batch = X_stream.iloc[start:end].copy()
    y_batch = y_stream.iloc[start:end].copy()

    if len(X_batch) > 0:
        batches.append((X_batch, y_batch))

print("\nTotal Stream Batches:", len(batches))


# ============================================================
# Initial Model
# ============================================================

print("\nTraining initial Random Forest model...")

current_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

current_model.fit(X_train, y_train)
current_model_name = "Random Forest"


# ============================================================
# Self-Healing Pipeline
# ============================================================

results = []
candidate_log = []

reference_data = X_train.copy()

X_train_dynamic = X_train.copy()
y_train_dynamic = y_train.copy()

for i, (X_batch_original, y_batch) in enumerate(batches, start=1):

    if i <= 5:
        drift_strength = 0.0
        X_batch = X_batch_original.copy()
    elif i <= 10:
        drift_strength = 0.3
        X_batch = simulate_data_drift(X_batch_original, drift_strength)
    else:
        drift_strength = 0.7
        X_batch = simulate_data_drift(X_batch_original, drift_strength)

    drift_score = calculate_kl_divergence(reference_data, X_batch)

    before_metrics = evaluate_model(current_model, X_batch, y_batch)
    before_f1 = before_metrics["F1-Score"]

    action, reason = agentic_decision_controller(before_f1, drift_score)

    selected_model_name = current_model_name
    after_f1 = before_f1
    healed = False

    print(
        f"\nBatch {i:02d} | Before F1: {before_f1:.4f} | "
        f"KL: {drift_score:.4f} | Agent Action: {action}"
    )

    if action == "Retrain":

        print("  Self-healing triggered: Retraining started...")

        X_train_dynamic = pd.concat([X_train_dynamic, X_batch], axis=0).reset_index(drop=True)
        y_train_dynamic = pd.concat([y_train_dynamic, y_batch], axis=0).reset_index(drop=True)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_dynamic,
            y_train_dynamic,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y_train_dynamic
        )

        best_model, best_model_name, best_f1, candidate_results = retrain_and_select_best(
            X_tr,
            y_tr,
            X_val,
            y_val
        )

        current_model = best_model
        current_model_name = best_model_name
        selected_model_name = best_model_name

        after_metrics = evaluate_model(current_model, X_batch, y_batch)
        after_f1 = after_metrics["F1-Score"]

        healed = after_f1 >= before_f1

        print(
            f"  Self-healing completed. Selected Model: {best_model_name} | "
            f"After F1: {after_f1:.4f}"
        )

        for c in candidate_results:
            c["Batch"] = i
            candidate_log.append(c)

    results.append({
        "Batch": i,
        "Drift_Strength": drift_strength,
        "KL-Drift-Score": drift_score,
        "Before_F1": before_f1,
        "Agent_Action": action,
        "Agent_Reason": reason,
        "Selected_Model": selected_model_name,
        "After_F1": after_f1,
        "Healed": healed
    })


# ============================================================
# Save Results
# ============================================================

results_df = pd.DataFrame(results)
results_df.to_csv("step4_self_healing_results.csv", index=False)

candidate_df = pd.DataFrame(candidate_log)
candidate_df.to_csv("step4_candidate_model_log.csv", index=False)

print("\nStep 4 Results Saved:")
print(" - step4_self_healing_results.csv")
print(" - step4_candidate_model_log.csv")

print("\nFinal Step 4 Results:")
print(results_df)

print("\nAgent Action Summary:")
print(results_df["Agent_Action"].value_counts())

print("\nSelf-Healing Events:")
print(results_df[results_df["Agent_Action"] == "Retrain"])