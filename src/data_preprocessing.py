from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

data_dir = BASE_DIR / "data"
data_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_dir / "machine_data.csv")

df.drop_duplicates(inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

df.to_csv(data_dir / "cleaned_data.csv", index=False)

print("cleaned_data.csv created")