import pandas as pd


# ============================================================
# Step 1: Load Step 2 Monitoring Results
# ============================================================

INPUT_FILE = "step2_drift_monitoring_results.csv"

df = pd.read_csv(INPUT_FILE)

print("\nLoaded Step 2 Monitoring Results")
print(df.head())


# ============================================================
# Step 2: Define Agentic Decision Thresholds
# ============================================================

F1_THRESHOLD = 0.70
KL_THRESHOLD = 0.05
CRITICAL_KL_THRESHOLD = 0.10


# ============================================================
# Step 3: Agentic Decision Controller
# ============================================================

def agentic_decision_controller(row):
    f1 = row["F1-Score"]
    kl = row["KL-Drift-Score"]

    if kl >= CRITICAL_KL_THRESHOLD:
        action = "Manual Review"
        reason = "Critical drift detected; human review recommended."

    elif f1 < F1_THRESHOLD and kl > KL_THRESHOLD:
        action = "Retrain"
        reason = "Performance degradation and significant drift detected."

    elif f1 < F1_THRESHOLD and kl <= KL_THRESHOLD:
        action = "Tune Hyperparameters"
        reason = "Performance degradation detected without significant drift."

    elif f1 >= F1_THRESHOLD and kl > KL_THRESHOLD:
        action = "Warning"
        reason = "Drift detected but model performance remains acceptable."

    else:
        action = "Continue"
        reason = "Model performance and drift score are within acceptable limits."

    return action, reason


# ============================================================
# Step 4: Apply Agent Decisions
# ============================================================

actions = []
reasons = []

for _, row in df.iterrows():
    action, reason = agentic_decision_controller(row)
    actions.append(action)
    reasons.append(reason)

df["Agent_Action"] = actions
df["Agent_Reason"] = reasons


# ============================================================
# Step 5: Print Agent Decision Log
# ============================================================

print("\nAgentic Decision Log")
print("============================================================")

for _, row in df.iterrows():
    print(
        f"Batch {int(row['Batch']):02d} | "
        f"F1: {row['F1-Score']:.4f} | "
        f"KL: {row['KL-Drift-Score']:.4f} | "
        f"Action: {row['Agent_Action']} | "
        f"Reason: {row['Agent_Reason']}"
    )


# ============================================================
# Step 6: Save Decision Results
# ============================================================

OUTPUT_FILE = "step3_agentic_decision_results.csv"
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nStep 3 Results Saved: {OUTPUT_FILE}")

print("\nAction Summary:")
print(df["Agent_Action"].value_counts())