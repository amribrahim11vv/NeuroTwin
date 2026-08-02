import pandas as pd
from pathlib import Path

BASE_DIR = Path("c:/Users/User/Desktop/Tribe V2 - PTSD Digital Brain Twin")
FILES = {
    "CI_clinical":   BASE_DIR / "CI_raw-data-scores.xlsx",
    "mood_stress":   BASE_DIR / "mood-stress-data.xlsx",
    "NF_datasheets": BASE_DIR / "NF_datasheets.xlsx",
    "MI_datasheets": BASE_DIR / "MI_datasheets.xlsx",
}

print("\n--- CI_clinical (raw-scores) ---")
df_ci = pd.read_excel(FILES["CI_clinical"], sheet_name="raw-scores", header=None, engine="openpyxl")
print("Shape:", df_ci.shape)
print("Rows 0 to 5, Cols 0 to 10:\n", df_ci.iloc[:10, :10])
print("Participant column unique values:\n", df_ci.iloc[2:, 0].dropna().unique())

print("\n--- mood_stress (mood-stress) ---")
df_mood = pd.read_excel(FILES["mood_stress"], sheet_name="mood-stress", header=None, engine="openpyxl")
print("Shape:", df_mood.shape)
print("Rows 0 to 8, Cols 0 to 10:\n", df_mood.iloc[:10, :10])

print("\n--- NF_datasheets (NF_summary-table) ---")
df_nf_sum = pd.read_excel(FILES["NF_datasheets"], sheet_name="NF_summary-table", nrows=5, engine="openpyxl")
print("NF_summary-table columns:", df_nf_sum.columns.tolist())
df_nf_used = pd.read_excel(FILES["NF_datasheets"], sheet_name="NF_used-data", nrows=5, engine="openpyxl")
print("NF_used-data columns:", df_nf_used.columns.tolist())
print("NF_used-data head:\n", df_nf_used)

print("\n--- MI_datasheets (DA_summary-table) ---")
df_mi_sum = pd.read_excel(FILES["MI_datasheets"], sheet_name="DA_summary-table", nrows=5, engine="openpyxl")
print("DA_summary-table columns:", df_mi_sum.columns.tolist())
print("DA_summary-table head:\n", df_mi_sum)
for sheet in ["theta_class-1", "alpha_class-1", "theta-alpha-ratio_class-1"]:
    df_s = pd.read_excel(FILES["MI_datasheets"], sheet_name=sheet, nrows=5, engine="openpyxl")
    print(f"\nMI sheet '{sheet}' columns:", df_s.columns.tolist())
    print("MI sheet head:\n", df_s)
