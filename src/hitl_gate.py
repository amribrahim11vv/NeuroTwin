"""
hitl_gate.py
=============
Phase 3  -  Human-in-the-Loop (HITL) Gate.

Presents each patient's RL recommendation to the clinician for
approval, override, or deferral. Logs all decisions to an audit trail.

Usage:
    python hitl_gate.py [--auto]   # --auto accepts all recommendations
"""

import pickle
import csv
import json
import sys
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).parent.parent
PKL_PATH    = BASE_DIR / "outputs" / "unified_patient_records.pkl"
Q_NET_PATH  = BASE_DIR / "models" / "cql_q_network.pt"
REPORTS_DIR = BASE_DIR / "outputs" / "clinical_reports"
AUDIT_LOG   = BASE_DIR / "outputs" / "hitl_audit_log.jsonl"

STATE_DIM  = 35
N_ACTIONS  = 3
ACTION_NAMES = ["CONTROL", "MI_SESSION", "NF_SESSION"]


class QNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 32),        nn.LayerNorm(32), nn.ReLU(),
            nn.Linear(32, n_actions),
        )
    def forward(self, s): return self.net(s)


def clean_vec(v, dim=STATE_DIM):
    out = [(x if (x is not None and not (isinstance(x, float) and x != x)) else 0.0) for x in (v or [])]
    out = out[:dim] + [0.0] * max(0, dim - len(out))
    return np.array(out, dtype=np.float32)


def safe(v):
    if v is None: return "N/A"
    if isinstance(v, float) and v != v: return "N/A"
    if isinstance(v, float): return f"{v:.2f}"
    return str(v)


def get_recommendation(q_net, state_vec):
    q_net.eval()
    s = torch.FloatTensor(clean_vec(state_vec)).unsqueeze(0)
    with torch.no_grad():
        q_vals = q_net(s).squeeze().numpy()
    best_idx = int(np.argmax(q_vals))
    return ACTION_NAMES[best_idx], q_vals


def log_decision(entry: dict):
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def run_hitl_review(records, q_net, auto_mode: bool = False):
    print("=" * 70)
    print("Tribe V2  -  Human-in-the-Loop Clinical Gate")
    print("=" * 70)
    if auto_mode:
        print("[MODE] Auto-accept  -  all RL recommendations will be approved")
    print()

    accepted, overridden, deferred = [], [], []

    for i, rec in enumerate(records):
        pid   = rec["participant_id"]
        group = rec["group"]
        sessions = rec.get("sessions", [])
        first_state = sessions[0].get("session_state_vec", []) if sessions else []

        rec_action, q_vals = get_recommendation(q_net, first_state)

        pre_pcl5  = safe(rec.get("pre_pcl5_total"))
        post_pcl5 = safe(rec.get("post_pcl5_total"))
        delta     = safe(rec.get("pcl5_delta"))

        print(f"[{i+1:02d}/{len(records)}] Patient: {pid}  |  Group: {group}")
        print(f"  PCL-5: {pre_pcl5} -> {post_pcl5}  (Delta = {delta})")
        print(f"  Sessions: {len(sessions)}")
        print(f"  RL Recommendation: > {rec_action}")
        print(f"  Q-values: Control={q_vals[0]:.4f}  MI={q_vals[1]:.4f}  NF={q_vals[2]:.4f}")

        if auto_mode:
            decision = "ACCEPT"
            override_action = rec_action
            clinician_note  = "Auto-accepted by system"
            print(f"  Decision: {decision}")
        else:
            print()
            print("  Options:")
            print("    [A] Accept recommendation")
            print("    [O] Override -> enter action (CONTROL / MI / NF)")
            print("    [D] Defer  -  flag for later review")
            print("    [S] Skip (no decision)")
            choice = input("  Your choice [A/O/D/S]: ").strip().upper()

            if choice == "A":
                decision = "ACCEPT"
                override_action = rec_action
                clinician_note = ""
            elif choice == "O":
                override_input = input("  Override to action [CONTROL/MI/NF]: ").strip().upper()
                action_map = {"CONTROL": "CONTROL", "MI": "MI_SESSION", "NF": "NF_SESSION"}
                override_action = action_map.get(override_input, rec_action)
                decision = "OVERRIDE"
                clinician_note = input("  Reason for override: ").strip()
            elif choice == "D":
                decision = "DEFER"
                override_action = None
                clinician_note = input("  Deferral note: ").strip()
            else:
                print("  Skipped.")
                print()
                continue

        # Audit log entry
        entry = {
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "participant_id":    pid,
            "group":             group,
            "rl_recommendation": rec_action,
            "q_values":          q_vals.tolist(),
            "decision":          decision,
            "final_action":      override_action,
            "clinician_note":    clinician_note if not auto_mode else "auto",
            "pre_pcl5":          rec.get("pre_pcl5_total"),
            "post_pcl5":         rec.get("post_pcl5_total"),
            "pcl5_delta":        rec.get("pcl5_delta"),
        }
        log_decision(entry)

        if decision == "ACCEPT":
            accepted.append(pid)
        elif decision == "OVERRIDE":
            overridden.append(pid)
        elif decision == "DEFER":
            deferred.append(pid)

        print(f"  -> Logged: {decision} | Final action: {override_action}")
        print()

    # Summary
    print("=" * 70)
    print("HITL SESSION SUMMARY")
    print("=" * 70)
    total = len(accepted) + len(overridden) + len(deferred)
    print(f"  Total reviewed:  {total}")
    print(f"  Accepted:        {len(accepted)}")
    print(f"  Overridden:      {len(overridden)}")
    print(f"  Deferred:        {len(deferred)}")
    if overridden:
        print(f"  Override PIDs:   {overridden}")
    if deferred:
        print(f"  Deferred PIDs:   {deferred}")
    print(f"\n  Audit log saved -> {AUDIT_LOG}")
    print("=" * 70)

    # Export CSV summary
    csv_path = REPORTS_DIR / "hitl_decisions.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if AUDIT_LOG.exists():
        rows = []
        with open(AUDIT_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line.strip()))
                except Exception:
                    pass
        if rows:
            keys = list(rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for row in rows:
                    row_flat = {k: (str(v) if isinstance(v, list) else v) for k, v in row.items()}
                    writer.writerow(row_flat)
            print(f"  Decisions CSV -> {csv_path}")


def main():
    auto_mode = "--auto" in sys.argv

    if not PKL_PATH.exists():
        print("[ERROR] unified_patient_records.pkl not found.")
        return

    with open(PKL_PATH, "rb") as f:
        records = pickle.load(f)

    if Q_NET_PATH.exists():
        try:
            ckpt  = torch.load(Q_NET_PATH, weights_only=True)
        except Exception:
            ckpt  = torch.load(Q_NET_PATH, weights_only=True)
        q_net = QNetwork()
        q_net.load_state_dict(ckpt["model_state_dict"])
    else:
        print("[WARN] Q-network not found  -  using random weights")
        q_net = QNetwork()

    run_hitl_review(records, q_net, auto_mode=auto_mode)


if __name__ == "__main__":
    main()
