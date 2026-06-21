import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "cleaned_data.csv"
OUTPUT_FILE = DATA_DIR / "engineered_data.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"File not found: {INPUT_FILE}\nRun data_preprocessing.py first."
    )

df = pd.read_csv(INPUT_FILE)

df["temp_pressure"] = df["temperature"] * df["pressure"]
df["vibration_temp"] = df["vibration"] * df["temperature"]

df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Saved: {OUTPUT_FILE}")

# Visualization
plt.figure(figsize=(8,5))
plt.scatter(df["temperature"], df["pressure"])
plt.xlabel("Temperature")
plt.ylabel("Pressure")
plt.title("Temperature vs Pressure")
plt.show()

plt.figure(figsize=(8,5))
df[["temp_pressure", "vibration_temp"]].hist(figsize=(10,4))
plt.tight_layout()
plt.show()