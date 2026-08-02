import pandas as pd
from pathlib import Path

path = Path("c:/Users/User/Desktop/Tribe V2 - PTSD Digital Brain Twin/CI_raw-data-scores.xlsx")
df = pd.read_excel(path, sheet_name="raw-scores", header=None, engine="openpyxl")

# Column 0 is participant, Column 1 is group
# Row 0 and 1 are headers, data starts at Row 3 (P01)
data = df.iloc[3:, [0, 1]]
data.columns = ["Participant", "Group"]
data["Group"] = data["Group"].ffill() # Forward fill group since some rows might have NaN group if they are grouped together
print(data)
