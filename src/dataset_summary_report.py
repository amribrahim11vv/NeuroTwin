"""
dataset_summary_report.py
==========================
Phase 1  -  Dataset Summary Report (script version).

Loads unified_patient_records.pkl and prints a comprehensive
summary of all 29 patient records.

Usage:
    python dataset_summary_report.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
PKL_PATH = BASE_DIR / "outputs" / "unified_patient_records.pkl"


def main():
    print("=" * 70)
    print("Tribe V2  -  Dataset Summary Report")
    print("=" * 70)

    if not PKL_PATH.exists():
        print(f"[ERROR] unified_patient_records.pkl not found.")
        print("        Please run data_ingestion_pipeline.py first.")
        return

    with open(PKL_PATH, "rb") as f:
        records = pickle.load(f)

    print(f"\n[LOAD] Total patient records: {len(records)}")

    # -- Group distribution -------------------------------------
    groups = {}
    for r in records:
        g = r.get("group", "UNKNOWN")
        groups[g] = groups.get(g, 0) + 1
    print("\n[GROUPS]")
    for g, n in sorted(groups.items()):
        print(f"  {g}: {n} participants")

    # -- Clinical score completeness ----------------------------
    clinical_cols = [
        "pre_pcl5_total", "post_pcl5_total", "pcl5_delta",
        "pre_wemwbs", "post_wemwbs",
        "pre_cd_risc", "post_cd_risc",
        "pre_brs", "post_brs", "pre_gse", "post_gse",
    ]
    print("\n[COMPLETENESS] Clinical scores:")
    for col in clinical_cols:
        vals = [r.get(col) for r in records]
        n_valid = sum(1 for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v)))
        print(f"  {col:<30s}: {n_valid}/{len(records)} valid")

    # -- PCL-5 statistics ---------------------------------------
    pre_pcl5  = [r.get("pre_pcl5_total") for r in records if
                 r.get("pre_pcl5_total") is not None and not np.isnan(r.get("pre_pcl5_total", float("nan")))]
    post_pcl5 = [r.get("post_pcl5_total") for r in records if
                 r.get("post_pcl5_total") is not None and not np.isnan(r.get("post_pcl5_total", float("nan")))]
    deltas    = [r.get("pcl5_delta") for r in records if
                 r.get("pcl5_delta") is not None and not np.isnan(r.get("pcl5_delta", float("nan")))]
    clinically_sig = sum(1 for r in records if r.get("pcl5_clinically_significant_pre", False))

    print(f"\n[PCL-5 STATISTICS]")
    if pre_pcl5:
        print(f"  Baseline (PRE) : mean={np.mean(pre_pcl5):.2f}, std={np.std(pre_pcl5):.2f}, "
              f"min={min(pre_pcl5):.1f}, max={max(pre_pcl5):.1f}")
    if post_pcl5:
        print(f"  Endline  (POST): mean={np.mean(post_pcl5):.2f}, std={np.std(post_pcl5):.2f}, "
              f"min={min(post_pcl5):.1f}, max={max(post_pcl5):.1f}")
    if deltas:
        improvers = sum(1 for d in deltas if d < 0)
        print(f"  PCL-5 Delta    : mean={np.mean(deltas):.2f}, std={np.std(deltas):.2f}")
        print(f"  Improvers (delta<0): {improvers}/{len(deltas)} ({100*improvers/len(deltas):.1f}%)")
    print(f"  Clinically significant at baseline (PCL-5 > 10): {clinically_sig}/{len(records)}")

    # -- Session counts -----------------------------------------
    print(f"\n[SESSION COUNTS PER PATIENT]")
    for r in records:
        pid = r.get("participant_id", "?")
        grp = r.get("group", "?")
        n_sess = len(r.get("sessions", []))
        imputed = sum(1 for s in r.get("sessions", []) if s.get("is_imputed", False))
        print(f"  {pid:<15s} [{grp:<8s}] -> {n_sess} sessions "
              f"({'imputed: '+str(imputed) if imputed else 'no imputation'})")

    # -- EEG feature availability -------------------------------
    print(f"\n[EEG FEATURE AVAILABILITY]")
    nf_eeg_counts, mi_eeg_counts = [], []
    for r in records:
        group = r.get("group", "")
        for s in r.get("sessions", []):
            vec = s.get("eeg_feature_vec", [])
            valid = sum(1 for v in vec if v is not None and not np.isnan(v))
            if group == "NF":
                nf_eeg_counts.append(valid / max(len(vec), 1))
            elif group == "MI":
                mi_eeg_counts.append(valid / max(len(vec), 1))

    if nf_eeg_counts:
        print(f"  NF group EEG completeness: {np.mean(nf_eeg_counts)*100:.1f}% avg")
    if mi_eeg_counts:
        print(f"  MI group EEG completeness: {np.mean(mi_eeg_counts)*100:.1f}% avg")

    # -- Sample record printout ---------------------------------
    print(f"\n[SAMPLE RECORDS]")
    for grp_target in ["NF", "MI", "CONTROL"]:
        sample = next((r for r in records if r.get("group") == grp_target), None)
        if sample:
            print(f"\n  --- {grp_target} sample ({sample['participant_id']}) ---")
            print(f"  pre_pcl5_total = {sample.get('pre_pcl5_total')}")
            print(f"  post_pcl5_total = {sample.get('post_pcl5_total')}")
            print(f"  pcl5_delta = {sample.get('pcl5_delta')}")
            print(f"  clinically_significant_pre = {sample.get('pcl5_clinically_significant_pre')}")
            print(f"  sessions = {len(sample.get('sessions', []))}")
            if sample.get("sessions"):
                s0 = sample["sessions"][0]
                print(f"  First session keys: {list(s0.keys())}")
                print(f"  eeg_feature_vec[:5] = {s0.get('eeg_feature_vec', [])[:5]}")

    print("\n[DONE] Summary complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
