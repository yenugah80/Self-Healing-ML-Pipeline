import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ============================================================
# Load Results
# ============================================================

baseline = pd.read_csv("step1_baseline_results.csv")
step4 = pd.read_csv("step4_self_healing_results.csv")

print("\nData Loaded Successfully")


# ============================================================
# 1. Baseline Model Comparison
# ============================================================

plt.figure(figsize=(10,6))
sns.barplot(data=baseline, x="Model", y="F1-Score")
plt.title("Baseline Model Comparison (F1 Score)")
plt.ylabel("F1 Score")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig("chart1_baseline_f1.png")
plt.show()


# ============================================================
# 2. F1 Score Over Time (Before vs After Healing)
# ============================================================

plt.figure(figsize=(12,6))

plt.plot(step4["Batch"], step4["Before_F1"], marker='o', label="Before Healing")
plt.plot(step4["Batch"], step4["After_F1"], marker='s', label="After Healing")

plt.title("F1 Score Over Time (Self-Healing Impact)")
plt.xlabel("Batch")
plt.ylabel("F1 Score")
plt.legend()
plt.tight_layout()
plt.savefig("chart2_f1_trend.png")
plt.show()


# ============================================================
# 3. Drift Score vs F1 Score
# ============================================================

plt.figure(figsize=(10,6))

sns.scatterplot(
    x=step4["KL-Drift-Score"],
    y=step4["Before_F1"],
    hue=step4["Agent_Action"],
    palette="Set1",
    s=100
)

plt.title("Drift vs Performance (Before Healing)")
plt.xlabel("KL Drift Score")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.savefig("chart3_drift_vs_f1.png")
plt.show()


# ============================================================
# 4. Agent Decision Distribution
# ============================================================

plt.figure(figsize=(8,6))

step4["Agent_Action"].value_counts().plot.pie(
    autopct='%1.1f%%',
    startangle=90
)

plt.title("Agent Decision Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig("chart4_agent_distribution.png")
plt.show()


# ============================================================
# 5. Self-Healing Improvement
# ============================================================

healing = step4[step4["Agent_Action"] == "Retrain"]

if len(healing) > 0:
    plt.figure(figsize=(8,6))

    plt.bar(healing["Batch"] - 0.2, healing["Before_F1"], width=0.4, label="Before")
    plt.bar(healing["Batch"] + 0.2, healing["After_F1"], width=0.4, label="After")

    plt.title("Self-Healing Effect (Before vs After)")
    plt.xlabel("Batch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("chart5_self_healing.png")
    plt.show()


# ============================================================
# 6. Summary Metrics
# ============================================================

print("\n================ FINAL SUMMARY ================")

avg_before = step4["Before_F1"].mean()
avg_after = step4["After_F1"].mean()

print(f"Average F1 Before Healing: {avg_before:.4f}")
print(f"Average F1 After Healing : {avg_after:.4f}")

improvement = avg_after - avg_before
print(f"Overall Improvement      : {improvement:.4f}")

print("\nSelf-Healing Success Rate:")
print(step4["Healed"].value_counts())

print("\nAgent Action Distribution:")
print(step4["Agent_Action"].value_counts())