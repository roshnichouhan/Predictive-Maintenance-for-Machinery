import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "tuned_model.pkl")

model = joblib.load(MODEL_PATH)


def predict_failure(machine_data):

    # ONLY FEATURES MODEL WAS TRAINED ON
    df = pd.DataFrame([{
        "temperature": machine_data["temperature"],
        "pressure": machine_data["pressure"],
        "speed": machine_data.get("speed", 1500),   # default if missing
        "vibration": machine_data["vibration"],

        "temp_pressure_ratio":
            machine_data["temperature"] / (machine_data["pressure"] + 1),

        "speed_vibration":
            machine_data.get("speed", 1500) * machine_data["vibration"]
    }])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    if probability > 0.7:
        status = "High Risk"
    elif probability > 0.4:
        status = "Medium Risk"
    else:
        status = "Low Risk"

    return {
        "failure_prediction": int(prediction),
        "failure_probability": round(float(probability), 3),
        "risk_level": status
    }


if __name__ == "__main__":

    sample = {
        "temperature": 90,
        "pressure": 75,
        "vibration": 6
    }

    result = predict_failure(sample)
    print(result)