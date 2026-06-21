import os
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# =========================
# 1. SAFE PATH SETTING
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "engineered_data.csv")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# reports folder create if not exists
os.makedirs(REPORT_DIR, exist_ok=True)

# =========================
# 2. LOAD MODEL
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# =========================
# 3. LOAD DATA
# =========================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("Columns in dataset:", df.columns)

# =========================
# 4. AUTO TARGET DETECTION
# =========================
possible_targets = ["target", "failure", "label", "machine_failure"]

target_col = None
for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    raise ValueError("Target column not found! Rename it to target/failure/label")

X = df.drop(target_col, axis=1)

# =========================
# 5. SHAP EXPLAINER
# =========================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# =========================
# 6. SUMMARY PLOT (GLOBAL)
# =========================
plt.figure()
shap.summary_plot(shap_values, X, show=False)

summary_path = os.path.join(REPORT_DIR, "shap_summary.png")
plt.savefig(summary_path, bbox_inches='tight')
plt.close()

# =========================
# 7. FORCE PLOT (SINGLE SAMPLE)
# =========================
shap.initjs()

force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X.iloc[0]
)

force_path = os.path.join(REPORT_DIR, "shap_force_plot.html")
shap.save_html(force_path, force_plot)

# =========================
# 8. DONE
# =========================
print("✅ SHAP explanation completed successfully!")
print(f"📊 Summary saved at: {summary_path}")
print(f"📄 Force plot saved at: {force_path}")