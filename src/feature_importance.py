import pandas as pd
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv("data/engineered_data.csv")

X = df.drop("failure", axis=1)

model = joblib.load("models/random_forest.pkl")

importance = model.feature_importances_

plt.barh(X.columns, importance)
plt.title("Feature Importance")
plt.show()