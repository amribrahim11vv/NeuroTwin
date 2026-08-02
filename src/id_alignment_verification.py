"""
id_alignment_verification.py
============================
Step 1 of Tribe V2 Phase 1  -  Foundation.

Loads all four source data files and audits participant IDs across them.
Produces a discrepancy report printed to the console and saved as
'id_alignment_report.txt'.

Usage:
    python id_alignment_verification.py
"""

import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# -- File paths --------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent

FILES = {
    "CI_clinical":   BASE_DIR / "data" / "CI_raw-data-scores.xlsx",
    "mood_stress":   BASE_DIR / "data" / "mood-stress-data.xlsx",
    "NF_datasheets": BASE_DIR / "data" / "NF_datasheets.xlsx",
    "MI_datasheets": BASE_DIR / "data" / "MI_datasheets.xlsx",
}


# -- Helpers -----------------------------------------------------------------
def get_all_sheet_names(path: Path) -> list[str]:
    xl = pd.ExcelFile(path, engine="openpyxl")
    return xl.sheet_names


def extract_ids_from_ci(path: Path) -> set[str]:
    """
    CI_raw-data-scores.xlsx has a complex multi-header layout.
    The participant ID is in column 0 starting from data row 2.
    """
    raw = pd.read_excel(path, sheet_name="raw-scores", header=None, engine="openpyxl")
    print(f"\n[CI] Sheet 'raw-scores' shape: {raw.shape}")
    print(f"[CI] First 4 rows of column 0:\n{raw.iloc[0:4, 0].to_list()}")
    
    # Data starts at row 2 (0-indexed). Column 0 is participant ID.
    ids = raw.iloc[2:, 0].dropna().astype(str).str.strip()
    ids = ids[ids != "nan"].unique().tolist()
    print(f"[CI] Found {len(ids)} participant IDs: {ids}")
    return set(ids)


def extract_ids_from_mood(path: Path) -> dict[str, set[str]]:
    """
    mood-stress-data.xlsx  -  extract participant IDs from column 0, normalising pXX -> PXX.
    """
    df = pd.read_excel(path, sheet_name="mood-stress", header=None, engine="openpyxl")
    raw_vals = df.iloc[4:, 0].dropna().astype(str).str.strip().tolist()
    
    # Filter for values starting with 'p' followed by digits
    ids = []
    for val in raw_vals:
        val_lower = val.lower()
        if val_lower.startswith("p") and val_lower[1:].isdigit():
            num = int(val_lower[1:])
            ids.append(f"P{num:02d}")
            
    print(f"[MOOD] Found and normalised {len(ids)} participant IDs: {sorted(set(ids))}")
    return {"mood-stress": set(ids)}


def extract_ids_from_eeg(label: str, path: Path) -> dict[str, set[str]]:
    """
    NF_datasheets.xlsx / MI_datasheets.xlsx  -  extract and normalise participant IDs to PXX.
    """
    if label == "NF":
        df = pd.read_excel(path, sheet_name="NF_summary-table", engine="openpyxl")
        sb_vals = pd.to_numeric(df["sb"], errors="coerce").dropna().astype(int).unique().tolist()
        ids = [f"P{val:02d}" for val in sb_vals if val in [16, 17, 18, 19, 20, 22, 23, 24, 25, 27]]
        print(f"[NF] Found and normalised {len(ids)} participant IDs from sb column: {sorted(ids)}")
        return {"NF_summary-table": set(ids)}
    
    elif label == "MI":
        df = pd.read_excel(path, sheet_name="DA_summary-table", engine="openpyxl")
        # Extract numbers from SubjID in original dataset, e.g. "MI_ 08" -> 8
        orig_ids = df["SubjID in original dataset"].dropna().astype(str).str.strip().unique().tolist()
        ids = []
        for orig in orig_ids:
            # remove "MI_" and spaces
            cleaned = orig.replace("MI_", "").replace(" ", "").strip()
            if cleaned.isdigit():
                num = int(cleaned)
                ids.append(f"P{num:02d}")
        print(f"[MI] Found and normalised {len(ids)} participant IDs from original dataset IDs: {sorted(ids)}")
        return {"DA_summary-table": set(ids)}
    
    return {}


def _find_id_column(df: pd.DataFrame) -> str | None:
    """
    Heuristic: find the first column whose name or values look like participant IDs.
    """
    # Check column names
    id_keywords = ["id", "participant", "subject", "code", "patient"]
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in id_keywords):
            return col
    # If no column name matches, look at first column values for pattern NF_XX / MI_XX / CTL_XX
    first_col = df.columns[0]
    sample = df[first_col].dropna().astype(str).head(5).tolist()
    if any(any(prefix in v for prefix in ["NF", "MI", "CTL", "Control", "NF_", "MI_"]) for v in sample):
        return first_col
    return None


# -- Main ---------------------------------------------------------------------
def main():
    lines = []
    log = lambda s: (print(s), lines.append(s))

    log("=" * 70)
    log("Tribe V2  -  Participant ID Alignment Verification")
    log("=" * 70)

    # 1. Load IDs from each source
    ci_ids = extract_ids_from_ci(FILES["CI_clinical"])
    mood_ids_by_sheet = extract_ids_from_mood(FILES["mood_stress"])
    nf_ids_by_sheet = extract_ids_from_eeg("NF", FILES["NF_datasheets"])
    mi_ids_by_sheet = extract_ids_from_eeg("MI", FILES["MI_datasheets"])

    # 2. Flatten multi-sheet dictionaries to single sets
    mood_ids = set().union(*mood_ids_by_sheet.values()) if mood_ids_by_sheet else set()
    nf_ids   = set().union(*nf_ids_by_sheet.values())   if nf_ids_by_sheet   else set()
    mi_ids   = set().union(*mi_ids_by_sheet.values())   if mi_ids_by_sheet   else set()

    all_sources = {
        "CI_clinical":   ci_ids,
        "mood_stress":   mood_ids,
        "NF_datasheets": nf_ids,
        "MI_datasheets": mi_ids,
    }

    # 3. Union of all IDs
    union_ids = set().union(*all_sources.values())
    
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"Total unique IDs across all files: {len(union_ids)}")
    for src, ids in all_sources.items():
        log(f"  {src}: {len(ids)} IDs -> {sorted(ids)}")

    # 4. Discrepancy report
    log("\n" + "=" * 70)
    log("DISCREPANCY REPORT")
    log("=" * 70)
    
    any_mismatch = False
    for src_a, ids_a in all_sources.items():
        for src_b, ids_b in all_sources.items():
            if src_a >= src_b:
                continue
            only_in_a = ids_a - ids_b
            only_in_b = ids_b - ids_a
            if only_in_a or only_in_b:
                any_mismatch = True
                log(f"\n[!] Mismatch between [{src_a}] and [{src_b}]:")
                if only_in_a:
                    log(f"   IDs only in {src_a}: {sorted(only_in_a)}")
                if only_in_b:
                    log(f"   IDs only in {src_b}: {sorted(only_in_b)}")

    if not any_mismatch:
        log("\n[OK] No ID mismatches detected across all four source files.")
    else:
        log("\n[!] Manual review required for mismatched IDs above before proceeding.")

    # 5. Save report
    report_path = BASE_DIR / "outputs" / "id_alignment_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[SAVED] Report written to: {report_path}")


if __name__ == "__main__":
    main()
