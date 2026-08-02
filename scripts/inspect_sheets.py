import pandas as pd
from pathlib import Path

BASE_DIR = Path("c:/Users/User/Desktop/Tribe V2 - PTSD Digital Brain Twin")
FILES = {
    "CI_clinical":   BASE_DIR / "CI_raw-data-scores.xlsx",
    "mood_stress":   BASE_DIR / "mood-stress-data.xlsx",
    "NF_datasheets": BASE_DIR / "NF_datasheets.xlsx",
    "MI_datasheets": BASE_DIR / "MI_datasheets.xlsx",
}

for name, path in FILES.items():
    if not path.exists():
        print(f"{name} does not exist at {path}")
        continue
    xl = pd.ExcelFile(path, engine="openpyxl")
    print(f"\n========================================\nFILE: {name}\n========================================")
    print("Sheets:", xl.sheet_names)
    for sheet in xl.sheet_names[:3]:
        df = pd.read_excel(path, sheet_name=sheet, nrows=5, engine="openpyxl")
        print(f"\nSheet: {sheet} (shape={df.shape if 'shape' in dir(df) else 'unknown'})")
        print("Columns:", df.columns.tolist()[:10])
        print("First row:\n", df.head(1))
