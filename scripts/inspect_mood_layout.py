import pandas as pd
from pathlib import Path

path = Path("c:/Users/User/Desktop/Tribe V2 - PTSD Digital Brain Twin/mood-stress-data.xlsx")
df = pd.read_excel(path, sheet_name="mood-stress", header=None, engine="openpyxl")
print("Shape of mood-stress sheet:", df.shape)
for r in range(10):
    row_vals = [str(x) for x in df.iloc[r].tolist()]
    print(f"Row {r}: {row_vals[:15]}")
