import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==========================
# PATH SETUP
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent


DATA_FILE = BASE_DIR / "data" / "engineered_data.csv"

MODEL_FILE = BASE_DIR / "models" / "xgboost_model.pkl"



# ==========================
# CHECK FILES
# ==========================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"{DATA_FILE} not found"
    )


if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"{MODEL_FILE} not found"
    )



# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(DATA_FILE)


print("Columns:")
print(df.columns)



# ==========================
# TARGET COLUMN
# ==========================

target_column = "failure"



X = df.drop(
    target_column,
    axis=1
)


y = df[target_column]



# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)



# ==========================
# LOAD MODEL
# ==========================

model = joblib.load(
    MODEL_FILE
)



# ==========================
# PREDICTION
# ==========================

y_pred = model.predict(
    X_test
)



if hasattr(model,"predict_proba"):

    y_prob = model.predict_proba(
        X_test
    )[:,1]

else:

    y_prob = y_pred



# ==========================
# METRICS
# ==========================


accuracy = accuracy_score(
    y_test,
    y_pred
)


precision = precision_score(
    y_test,
    y_pred
)


recall = recall_score(
    y_test,
    y_pred
)


f1 = f1_score(
    y_test,
    y_pred
)


roc = roc_auc_score(
    y_test,
    y_prob
)



# ==========================
# RESULT
# ==========================

print("\n======================")
print(" MODEL PERFORMANCE ")
print("======================")


print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc:.4f}"
)



print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred
    )
)



print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)