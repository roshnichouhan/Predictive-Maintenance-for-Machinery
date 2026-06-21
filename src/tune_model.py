import pandas as pd
import joblib
import os


from sklearn.model_selection import (
    GridSearchCV,
    train_test_split
)


from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)



# =====================================
# PROJECT ROOT PATH
# =====================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)



DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "engineered_data.csv"
)



MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_rf.pkl"
)



# =====================================
# CHECK DATA FILE
# =====================================

if not os.path.exists(DATA_PATH):

    print(
        "❌ File not found:",
        DATA_PATH
    )

    raise FileNotFoundError(
        "engineered_data.csv missing. Run feature_engineering.py first"
    )



# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv(
    DATA_PATH
)



print("\nDataset Columns:")
print(df.columns)



# =====================================
# TARGET COLUMN
# =====================================

if "failure" not in df.columns:

    raise ValueError(
        "❌ failure column not found"
    )



X = df.drop(
    "failure",
    axis=1
)


y = df["failure"]



# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)



# =====================================
# PARAMETER GRID
# =====================================

params = {


    "n_estimators":[

        100,
        200,
        300

    ],


    "max_depth":[

        5,
        10,
        20

    ]

}



# =====================================
# RANDOM FOREST MODEL
# =====================================


rf = RandomForestClassifier(

    random_state=42

)



grid = GridSearchCV(

    estimator=rf,

    param_grid=params,

    cv=5,

    scoring="f1",

    n_jobs=-1,

    verbose=1

)



# =====================================
# TRAIN
# =====================================

print("\n🚀 Training started...")


grid.fit(

    X_train,

    y_train

)



# =====================================
# TEST PERFORMANCE
# =====================================


best_model = grid.best_estimator_



prediction = best_model.predict(

    X_test

)



accuracy = accuracy_score(

    y_test,

    prediction

)


precision = precision_score(

    y_test,

    prediction

)


recall = recall_score(

    y_test,

    prediction

)


f1 = f1_score(

    y_test,

    prediction

)



print("\n====================")
print("BEST MODEL RESULT")
print("====================")


print(
    "Best Params:",
    grid.best_params_
)



print(
    f"Accuracy : {accuracy:.4f}"
)


print(
    f"Precision: {precision:.4f}"
)


print(
    f"Recall   : {recall:.4f}"
)


print(
    f"F1 Score : {f1:.4f}"
)



# =====================================
# SAVE MODEL
# =====================================


os.makedirs(

    os.path.dirname(MODEL_PATH),

    exist_ok=True

)



joblib.dump(

    best_model,

    MODEL_PATH

)



print(
    "\n✅ Best model saved:"
)

print(
    MODEL_PATH
)