import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

BASE = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE, "..", "data", "engineered_data.csv")
MODEL_DIR = os.path.join(BASE, "..", "models")

def train():

    if not os.path.exists(DATA_PATH):
        raise Exception("❌ engineered_data.csv missing. Run feature_engineering.py first")

    df = pd.read_csv(DATA_PATH)

    X = df.drop("failure", axis=1)
    y = df["failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    os.makedirs(MODEL_DIR, exist_ok=True)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))
    print("✅ Random Forest Saved")

    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        eval_metric="logloss"
    )

    xgb.fit(X_train, y_train)

    joblib.dump(xgb, os.path.join(MODEL_DIR, "xgboost_model.pkl"))
    print("✅ XGBoost Saved")

if __name__ == "__main__":
    train()