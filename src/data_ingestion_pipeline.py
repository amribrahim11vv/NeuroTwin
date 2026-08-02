"""
data_ingestion_pipeline.py
===========================
Step 2 of Tribe V2 Phase 1  -  Foundation.

Parses all four source files and produces a unified list of PatientRecord
dictionaries, saved as 'unified_patient_records.pkl'.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from copy import deepcopy

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
FILES = {
    "CI":   BASE_DIR / "data" / "CI_raw-data-scores.xlsx",
    "MOOD": BASE_DIR / "data" / "mood-stress-data.xlsx",
    "NF":   BASE_DIR / "data" / "NF_datasheets.xlsx",
    "MI":   BASE_DIR / "data" / "MI_datasheets.xlsx",
}

# -------------------------------------------------------------
# STEP 1: Parse CI_raw-data-scores.xlsx (multi-header layout)
# -------------------------------------------------------------
def parse_ci_scores(path: Path) -> list[dict]:
    print("\n[CI] Loading raw-scores sheet...")
    raw = pd.read_excel(path, sheet_name="raw-scores", header=None, engine="openpyxl")
    
    # Build multi-index from row 0 (block labels) and row 1 (item labels)
    block_row = raw.iloc[0].ffill()
    item_row  = raw.iloc[1]
    multi_idx = pd.MultiIndex.from_arrays([block_row, item_row])

    df = raw.iloc[2:].copy().reset_index(drop=True)
    df.columns = multi_idx

    records = []
    current_group = "CONTROL"
    
    for idx, row in df.iterrows():
        pid = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else None
        if not pid or pid.lower() in ("nan", "participant", ""):
            continue

        # Forward-fill group label
        group_val = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else None
        if group_val and group_val != "nan":
            group_upper = group_val.upper()
            if "CONTROL" in group_upper:
                current_group = "CONTROL"
            elif "BCI" in group_upper:
                current_group = "MI"
            elif "NEUROFEEDBACK" in group_upper or "NF" in group_upper:
                current_group = "NF"
        
        group = current_group

        def find_score(df_row, block_keyword, item_keyword):
            for (blk, itm), val in df_row.items():
                if block_keyword.lower() in str(blk).lower() and item_keyword.lower() in str(itm).lower():
                    try:
                        return float(val)
                    except Exception:
                        continue
            return np.nan

        rec = {
            "participant_id": pid,
            "group": group,
            "pre_pcl5_total":      find_score(row, "baseline", "pcl"),
            "pre_pcl5_intrusion":  find_score(row, "baseline", "intrusion"),
            "pre_pcl5_avoidance":  find_score(row, "baseline", "avoidance"),
            "pre_pcl5_anac":       find_score(row, "baseline", "anac"),
            "pre_pcl5_arousal":    find_score(row, "baseline", "arousal"),
            "pre_pc_ptsd":         find_score(row, "baseline", "pc-ptsd"),
            "pre_htq":             find_score(row, "baseline", "htq"),
            "pre_wemwbs":          find_score(row, "baseline", "wemwbs"),
            "pre_cd_risc":         find_score(row, "baseline", "cd-risc"),
            "pre_brs":             find_score(row, "baseline", "brs"),
            "pre_gse":             find_score(row, "baseline", "gse"),
            "post_pcl5_total":     find_score(row, "endline", "pcl"),
            "post_pcl5_intrusion": find_score(row, "endline", "intrusion"),
            "post_pcl5_avoidance": find_score(row, "endline", "avoidance"),
            "post_pcl5_anac":      find_score(row, "endline", "anac"),
            "post_pcl5_arousal":   find_score(row, "endline", "arousal"),
            "post_pc_ptsd":        find_score(row, "endline", "pc-ptsd"),
            "post_htq":            find_score(row, "endline", "htq"),
            "post_wemwbs":         find_score(row, "endline", "wemwbs"),
            "post_cd_risc":        find_score(row, "endline", "cd-risc"),
            "post_brs":            find_score(row, "endline", "brs"),
            "post_gse":            find_score(row, "endline", "gse"),
        }
        
        pre = rec["pre_pcl5_total"]
        post = rec["post_pcl5_total"]
        rec["pcl5_delta"] = (post - pre) if (pd.notna(pre) and pd.notna(post)) else np.nan
        rec["pcl5_clinically_significant_pre"] = bool(pd.notna(pre) and pre > 10)
        rec["sessions"] = []

        records.append(rec)

    print(f"[CI] Parsed {len(records)} participants.")
    return records


# -------------------------------------------------------------
# STEP 2: Parse mood-stress-data.xlsx
# -------------------------------------------------------------
def parse_mood_stress(path: Path) -> dict[str, list[dict]]:
    print("\n[MOOD] Loading mood-stress data...")
    df = pd.read_excel(path, sheet_name="mood-stress", header=None, engine="openpyxl")
    
    all_sessions = {}
    
    for idx in range(5, df.shape[0]):
        row = df.iloc[idx]
        pid_raw = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else None
        if not pid_raw or pid_raw.lower() in ("nan", "nf", "bci", ""):
            continue
        
        # Normalise pid: p08 -> P08
        if pid_raw.lower().startswith("p") and pid_raw[1:].isdigit():
            pid = f"P{int(pid_raw[1:]):02d}"
        else:
            continue
            
        sessions_list = []
        for s in range(1, 8):
            col_idx = 2 + 5 * (s - 1)
            
            def safe_float(v):
                try:
                    f = float(v)
                    return f if not np.isnan(f) else np.nan
                except:
                    return np.nan
            
            mp   = safe_float(row.iloc[col_idx])
            sp   = safe_float(row.iloc[col_idx + 1])
            mpo  = safe_float(row.iloc[col_idx + 2])
            spo  = safe_float(row.iloc[col_idx + 3])
            
            if np.isnan(mp) and np.isnan(sp) and np.isnan(mpo) and np.isnan(spo):
                continue
                
            sess = {
                "session_index": s,
                "mood_pre":    mp,
                "mood_post":   mpo,
                "stress_pre":  sp,
                "stress_post": spo,
                "mood_delta":  (mpo - mp) if (pd.notna(mpo) and pd.notna(mp)) else np.nan,
                "stress_delta": (spo - sp) if (pd.notna(spo) and pd.notna(sp)) else np.nan,
            }
            sessions_list.append(sess)
            
        all_sessions[pid] = sessions_list
        
    print(f"[MOOD] Found mood/stress data for {len(all_sessions)} participants.")
    return all_sessions


# -------------------------------------------------------------
# STEP 3: Parse NF_datasheets.xlsx
# -------------------------------------------------------------
def parse_nf_data(path: Path) -> dict[str, list[dict]]:
    print("\n[NF] Loading NF datasheets...")
    df = pd.read_excel(path, sheet_name="NF_summary-table", engine="openpyxl")
    
    # Clean sb column to get numeric IDs
    df["sb"] = pd.to_numeric(df["sb"], errors="coerce")
    df = df.dropna(subset=["sb"])
    df["sb"] = df["sb"].astype(int)
    df["ss"] = pd.to_numeric(df["ss"], errors="coerce").fillna(1).astype(int)

    all_nf = {}
    
    def safe_float(v):
        if pd.isna(v):
            return np.nan
        try:
            v_str = str(v).strip()
            if v_str in ('[]', '', 'nan', 'None'):
                return np.nan
            return float(v)
        except:
            return np.nan
    
    for _, row in df.iterrows():
        sb = int(row["sb"])
        if sb not in [16, 17, 18, 19, 20, 22, 23, 24, 25, 27]:
            continue
            
        pid = f"P{sb:02d}"
        sess_idx = int(row["ss"])
        
        # EEG Gravity Bands
        allch = [safe_float(row[f"allCh_G{i}"]) if f"allCh_G{i}" in row else np.nan for i in range(1, 11)]
        onech = [safe_float(row[f"oneCh_G{i}"]) if f"oneCh_G{i}" in row else np.nan for i in range(1, 11)]
        
        sess = {
            "session_index": sess_idx,
            "eeg_allch": allch,
            "eeg_onech": onech,
            "r1_trial_sec": safe_float(row["R1_tr_sec"]) if "R1_tr_sec" in row else np.nan,
            "g0_trial_sec": safe_float(row["G0_tr_sec"]) if "G0_tr_sec" in row else np.nan,
        }
        
        if pid not in all_nf:
            all_nf[pid] = []
        all_nf[pid].append(sess)
        
    print(f"[NF] Found NF EEG data for {len(all_nf)} participants.")
    return all_nf


# -------------------------------------------------------------
# STEP 4: Parse MI_datasheets.xlsx
# -------------------------------------------------------------
def parse_mi_data(path: Path) -> dict[str, list[dict]]:
    print("\n[MI] Loading MI datasheets...")
    xl = pd.ExcelFile(path, engine="openpyxl")
    
    # Load classifiers
    df_sum = pd.read_excel(xl, "DA_summary-table", engine="openpyxl")
    df_1s  = pd.read_excel(xl, "DA_1-sec-classification-window", engine="openpyxl")
    df_2s  = pd.read_excel(xl, "DA_2-sec-classification-window", engine="openpyxl")
    
    # Load spectral features matrices (subjects 0-9, sessions 0-5)
    theta_m = pd.read_excel(xl, "theta_class-1", header=None, engine="openpyxl").values
    alpha_m = pd.read_excel(xl, "alpha_class-2", header=None, engine="openpyxl").values
    ratio_m = pd.read_excel(xl, "theta-alpha-ratio_both-classes", header=None, engine="openpyxl").values
    
    # Subject mappings to row index
    sub_map = {
        8: 0, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6, 15: 7, 26: 8, 28: 9
    }
    
    all_mi = {}
    
    for sb_id, sub_idx in sub_map.items():
        pid = f"P{sb_id:02d}"
        sessions_list = []
        
        for s in range(1, 7):
            # Find runs matching this subject and session
            sub_runs = df_sum[(df_sum["SubjID for paper"] == (sub_idx + 1)) & (df_sum["Session"] == s)]
            if sub_runs.empty:
                continue
                
            run_indices = sub_runs.index.tolist()
            
            # Average accuracy over runs
            acc_1s_vals = [float(df_1s.iloc[idx]["ORIG_refPeak_DA_mean"]) for idx in run_indices if pd.notna(df_1s.iloc[idx]["ORIG_refPeak_DA_mean"])]
            acc_2s_vals = [float(df_2s.iloc[idx]["ORIG_refPeak_DA_mean"]) for idx in run_indices if pd.notna(df_2s.iloc[idx]["ORIG_refPeak_DA_mean"])]
            
            da_1 = np.mean(acc_1s_vals) if acc_1s_vals else np.nan
            da_2 = np.mean(acc_2s_vals) if acc_2s_vals else np.nan
            
            # Spectral features from matrix
            theta = float(theta_m[sub_idx, s - 1]) if sub_idx < theta_m.shape[0] and (s-1) < theta_m.shape[1] else np.nan
            alpha = float(alpha_m[sub_idx, s - 1]) if sub_idx < alpha_m.shape[0] and (s-1) < alpha_m.shape[1] else np.nan
            ratio = float(ratio_m[sub_idx, s - 1]) if sub_idx < ratio_m.shape[0] and (s-1) < ratio_m.shape[1] else np.nan
            
            sess = {
                "session_index": s,
                "da_1sec_accuracy": da_1,
                "da_2sec_accuracy": da_2,
                "theta_class1":     theta,
                "alpha_class2":     alpha,
                "theta_alpha_ratio": ratio,
            }
            sessions_list.append(sess)
            
        all_mi[pid] = sessions_list
        
    print(f"[MI] Found MI EEG data for {len(all_mi)} participants.")
    return all_mi


# -------------------------------------------------------------
# STEP 5: Merge into unified PatientRecord list
# -------------------------------------------------------------
def merge_records(ci_records, mood_data, nf_data, mi_data) -> list[dict]:
    unified = []

    for rec in ci_records:
        pid = rec["participant_id"]
        group = rec["group"]
        result = deepcopy(rec)

        mood_sessions = mood_data.get(pid, [])
        nf_sessions   = nf_data.get(pid, [])
        mi_sessions   = mi_data.get(pid, [])

        session_indices = sorted(set(
            [s["session_index"] for s in mood_sessions] +
            [s["session_index"] for s in nf_sessions] +
            [s["session_index"] for s in mi_sessions]
        ))

        if not session_indices:
            session_indices = [0]

        sessions = []
        for si in session_indices:
            mood_s = next((s for s in mood_sessions if s["session_index"] == si), {})
            nf_s   = next((s for s in nf_sessions   if s["session_index"] == si), {})
            mi_s   = next((s for s in mi_sessions   if s["session_index"] == si), {})

            sess = {
                "session_index": si,
                "mood_pre":    mood_s.get("mood_pre",    np.nan),
                "mood_post":   mood_s.get("mood_post",   np.nan),
                "stress_pre":  mood_s.get("stress_pre",  np.nan),
                "stress_post": mood_s.get("stress_post", np.nan),
                "mood_delta":  mood_s.get("mood_delta",  np.nan),
                "stress_delta":mood_s.get("stress_delta",np.nan),
                # NF
                "eeg_allch":   nf_s.get("eeg_allch",  [np.nan]*10),
                "eeg_onech":   nf_s.get("eeg_onech",  [np.nan]*10),
                "r1_trial_sec":nf_s.get("r1_trial_sec", np.nan),
                "g0_trial_sec":nf_s.get("g0_trial_sec", np.nan),
                # MI
                "da_1sec_accuracy":  mi_s.get("da_1sec_accuracy",  np.nan),
                "da_2sec_accuracy":  mi_s.get("da_2sec_accuracy",  np.nan),
                "theta_class1":      mi_s.get("theta_class1",      np.nan),
                "alpha_class2":      mi_s.get("alpha_class2",      np.nan),
                "theta_alpha_ratio": mi_s.get("theta_alpha_ratio", np.nan),
                "is_imputed": False,
            }

            # Build EEG feature vector
            if group == "NF":
                eeg_feat = sess["eeg_allch"] + sess["eeg_onech"] + \
                           [sess["r1_trial_sec"], sess["g0_trial_sec"]]
            elif group == "MI":
                eeg_feat = [sess["theta_class1"], sess["alpha_class2"],
                            sess["theta_alpha_ratio"], sess["da_1sec_accuracy"],
                            sess["da_2sec_accuracy"]]
            else:  # CONTROL
                eeg_feat = [np.nan] * 5

            # Pad / format features to match STATE_DIM = 35
            # State vec size: eeg_feat + 2 mood/stress + 3 clinical baseline = 35
            # NF: 10 + 10 + 2 + 5 (mood/stress/clinical) = 27 (pad to 35)
            # MI: 5 + 5 = 10 (pad to 35)
            state_vec = eeg_feat + [
                sess["mood_pre"],
                sess["stress_pre"],
                rec.get("pre_pcl5_total", np.nan),
                rec.get("pre_wemwbs",     np.nan),
                rec.get("pre_cd_risc",    np.nan),
            ]
            
            # Align state vector size to 35
            state_vec += [0.0] * max(0, 35 - len(state_vec))
            
            sess["eeg_feature_vec"] = eeg_feat
            sess["session_state_vec"] = state_vec
            sessions.append(sess)

        sessions = impute_sessions(sessions, group)
        result["sessions"] = sessions
        unified.append(result)

    return unified


def impute_sessions(sessions: list[dict], group: str) -> list[dict]:
    if group == "CONTROL":
        return sessions

    scalar_fields = ["r1_trial_sec", "g0_trial_sec", "da_1sec_accuracy",
                     "da_2sec_accuracy", "theta_class1", "alpha_class2",
                     "theta_alpha_ratio"]

    for field in scalar_fields:
        values = [s[field] for s in sessions if pd.notna(s[field])]
        if not values:
            continue
        mean_val = np.mean(values)
        for s in sessions:
            if pd.isna(s[field]):
                s[field] = mean_val
                s["is_imputed"] = True

    return sessions


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():
    print("=" * 70)
    print("Tribe V2  -  Data Ingestion Pipeline")
    print("=" * 70)

    ci_records = parse_ci_scores(FILES["CI"])
    mood_data  = parse_mood_stress(FILES["MOOD"])
    nf_data    = parse_nf_data(FILES["NF"])
    mi_data    = parse_mi_data(FILES["MI"])

    unified = merge_records(ci_records, mood_data, nf_data, mi_data)

    out_path = BASE_DIR / "outputs" / "unified_patient_records.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(unified, f)

    print("\n" + "=" * 70)
    print(f"[DONE] Saved {len(unified)} patient records to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
