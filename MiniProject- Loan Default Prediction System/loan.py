import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11})

# ==========================================
# 1. Load Data & Exploratory Data Analysis
# ==========================================
df = pd.read_csv(r"C:\Users\rouna\Downloads\Module 3 Project Data.csv")

# Drop non-predictive Index column
if "Index" in df.columns:
    df = df.drop(columns=["Index"])

print("Dataset Overview:")
print(df.info())
print("\nTarget Class Distribution (%):")
print((df["Defaulted?"].value_counts(normalize=True) * 100).round(2))

# Figure 1: Class Distribution
plt.figure(figsize=(6, 5))
palette = ["#4C72B0", "#C44E52"]
ax = sns.countplot(
    x="Defaulted?",
    hue="Defaulted?",
    data=df,
    palette=palette,
    legend=False
)
plt.title("Class Distribution: Loan Default", fontsize=14, pad=12)
plt.xlabel("Defaulted (0 = No, 1 = Yes)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.show()

# Figure 2: Correlation Heatmap
plt.figure(figsize=(7, 6))
corr_matrix = df.corr()
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-0.2,
    vmax=1.0,
    square=True,
    cbar_kws={"shrink": 0.8}
)
plt.title("Feature Correlation Heatmap", fontsize=14, pad=12)
plt.tight_layout()
plt.show()

# ==========================================
# 2. Data Preprocessing & Train/Test Split
# ==========================================
X = df.drop(columns=["Defaulted?"])
y = df["Defaulted?"]

# 80/20 Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Standardize features for scale-sensitive models (Logistic Regression & SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate class imbalance ratio for XGBoost scale_pos_weight (~29)
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

# ==========================================
# 3. Model Definition & Training
# ==========================================
models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "use_scaled": True
    },
    "Random Forest": {
        "model": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
        "use_scaled": False
    },
    "SVM (RBF)": {
        "model": SVC(probability=True, class_weight="balanced", random_state=42),
        "use_scaled": True
    },
    "XGBoost": {
        "model": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss"
        ),
        "use_scaled": False
    }
}

results = []
roc_curves = {}

for name, config in models.items():
    clf = config["model"]
    is_scaled = config["use_scaled"]
    
    # Train
    X_tr = X_train_scaled if is_scaled else X_train
    X_te = X_test_scaled if is_scaled else X_test
    clf.fit(X_tr, y_train)
    
    # Predict
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]
    
    # Store ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_curves[name] = (fpr, tpr)
    
    # Compute metrics
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

# Convert results to DataFrame and sort by ROC-AUC
results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
print("\n" + "=" * 65)
print("Table 1: Test-Set Performance Comparison")
print("=" * 65)
print(results_df.to_string(index=False, float_format="{:.3f}".format))

# ==========================================
# 4. Evaluation Visualizations
# ==========================================

# Figure 3: Model Performance Comparison Bar Chart
metrics_melted = results_df.melt(
    id_vars="Model",
    value_vars=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=metrics_melted,
    x="Model",
    y="Score",
    hue="Metric",
    palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
)
plt.title("Model Performance Comparison", fontsize=14, pad=12)
plt.ylim(0, 1.05)
plt.ylabel("Score", fontsize=12)
plt.xlabel("")
plt.legend(loc="lower right", frameon=True)
plt.tight_layout()
plt.show()

# Figure 4: ROC Curves
plt.figure(figsize=(8, 7))
for name in results_df["Model"]:
    fpr, tpr = roc_curves[name]
    auc = results_df.loc[results_df["Model"] == name, "ROC-AUC"].values[0]
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", lw=2)

plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance Level")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves — Model Comparison", fontsize=14, pad=12)
plt.legend(loc="lower right", frameon=True)
plt.tight_layout()
plt.show()

# Figure 5: Confusion Matrix for Logistic Regression
lr_model = models["Logistic Regression"]["model"]
lr_pred = lr_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, lr_pred)

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Default", "Default"])
disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=True)
plt.title("Confusion Matrix — Logistic Regression", fontsize=14, pad=12)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.show()

# ==========================================
# 5. Feature Importance Analysis
# ==========================================
rf_importances = models["Random Forest"]["model"].feature_importances_
xgb_importances = models["XGBoost"]["model"].feature_importances_
features = X.columns
feature_order = ["Bank Balance", "Annual Salary", "Employed"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Random Forest Feature Importance
sns.barplot(
    x=rf_importances,
    y=features,
    hue=features,
    ax=axes[0],
    palette="viridis",
    order=feature_order,
    legend=False
)
axes[0].set_title("Feature Importance — Random Forest", fontsize=13)
axes[0].set_xlabel("Importance", fontsize=11)
axes[0].set_ylabel("")

# XGBoost Feature Importance
sns.barplot(
    x=xgb_importances,
    y=features,
    hue=features,
    ax=axes[1],
    palette="magma",
    order=feature_order,
    legend=False
)
axes[1].set_title("Feature Importance — XGBoost", fontsize=13)
axes[1].set_xlabel("Importance", fontsize=11)
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()