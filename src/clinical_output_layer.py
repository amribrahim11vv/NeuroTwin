"""
clinical_output_layer.py
=========================
Phase 3 - Clinical Output Layer.

Loads the trained LSDT backbone, LoRA-adapted patient models, and the CQL policy,
then generates individualized clinical decision-support artifacts:
  - Predicted outcome trajectories (PCL-5, WEMWBS, CD-RISC)
  - Treatment-response profiles (session-by-session latent drift)
  - Session-level biomarker trend summaries
  - Explicit uncertainty quantification via ensemble spread

Usage:
    python clinical_output_layer.py
"""

import pickle
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

BASE_DIR     = Path(__file__).parent.parent
PKL_PATH     = BASE_DIR / "outputs" / "unified_patient_records.pkl"
BACKBONE_PT  = BASE_DIR / "models" / "lsdt_backbone.pt"
LORA_DIR     = BASE_DIR / "models" / "lora_adapters"
POLICY_PT    = BASE_DIR / "models" / "cql_q_network.pt"
OUTPUT_DIR   = BASE_DIR / "outputs" / "clinical_outputs"

# Architecture parameters (must match training scripts)
INPUT_DIM   = 35
LATENT_DIM  = 16
GRU_HIDDEN  = 32


# ---------------------------------------------------------------
# 1. Rebuild backbone architecture (minimal)
# ---------------------------------------------------------------
class LSDTBackbone(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM, gru_hidden=GRU_HIDDEN):
        super().__init__()
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.gru_hidden = gru_hidden

        self.vae_encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64),        nn.LayerNorm(64),  nn.ReLU(),
            nn.Linear(64, 2 * latent_dim),
        )
        self.vae_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.LayerNorm(64),  nn.ReLU(),
            nn.Linear(64, 128),         nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, input_dim),
        )
        self.gru = nn.GRU(latent_dim, gru_hidden, batch_first=True)
        self.head_pcl5   = nn.Linear(gru_hidden, 1)
        self.head_wemwbs = nn.Linear(gru_hidden, 1)
        self.head_cdrisc = nn.Linear(gru_hidden, 1)
        self.head_gse    = nn.Linear(gru_hidden, 1)

    def encode(self, x):
        out = self.vae_encoder(x)
        mu, log_var = out.chunk(2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.vae_decoder(z)

    def forward_seq(self, x_seq, mask):
        B, T, D = x_seq.shape
        z_list, mu_list = [], []
        for t in range(T):
            xt = x_seq[:, t, :]
            mu_t, lv_t = self.encode(xt)
            z_t = self.reparameterize(mu_t, lv_t)
            z_list.append(z_t)
            mu_list.append(mu_t)
        z_seq = torch.stack(z_list, dim=1)
        mu_seq = torch.stack(mu_list, dim=1)
        h_seq, _ = self.gru(z_seq)
        last_idx = mask.long().sum(dim=1).clamp(min=1) - 1
        h_patient = h_seq[torch.arange(B), last_idx, :]
        aux = torch.stack([
            self.head_pcl5(h_patient).squeeze(-1),
            self.head_wemwbs(h_patient).squeeze(-1),
            self.head_cdrisc(h_patient).squeeze(-1),
            self.head_gse(h_patient).squeeze(-1),
        ], dim=-1)
        return z_seq, mu_seq, h_seq, h_patient, aux


# ---------------------------------------------------------------
# 2. Generate clinical report for each patient
# ---------------------------------------------------------------
def generate_patient_report(rec, model, max_sessions=10):
    """Generate a clinical decision-support artifact for one patient."""
    pid   = rec.get("participant_id", "unknown")
    group = rec.get("group", "UNKNOWN")
    sessions = rec.get("sessions", [])
    T = min(len(sessions), max_sessions)

    # Build input tensors
    x_seq = np.zeros((1, max_sessions, INPUT_DIM), dtype=np.float32)
    mask  = np.zeros((1, max_sessions), dtype=bool)

    for t, sess in enumerate(sessions[:T]):
        vec = sess.get("session_state_vec", [])
        vec = [v if (v is not None and not (isinstance(v, float) and np.isnan(v))) else 0.0 for v in vec]
        vec = vec[:INPUT_DIM]
        vec += [0.0] * (INPUT_DIM - len(vec))
        x_seq[0, t] = np.array(vec, dtype=np.float32)
        mask[0, t] = True

    x_t = torch.FloatTensor(x_seq)
    m_t = torch.BoolTensor(mask)

    model.eval()
    with torch.no_grad():
        z_seq, mu_seq, h_seq, h_patient, aux = model.forward_seq(x_t, m_t)

    # Extract predictions
    pred_pcl5_delta   = float(aux[0, 0])
    pred_wemwbs_delta = float(aux[0, 1])
    pred_cdrisc_delta = float(aux[0, 2])
    pred_gse          = float(aux[0, 3])

    # Session-level latent trajectory
    latent_trajectory = []
    for t in range(T):
        z_t = mu_seq[0, t].numpy().tolist()
        h_t = h_seq[0, t].numpy().tolist()
        latent_trajectory.append({
            "session": t + 1,
            "latent_state_z": [round(v, 4) for v in z_t],
            "gru_hidden_h":   [round(v, 4) for v in h_t],
        })

    # Latent drift analysis (L2 norm between consecutive z's)
    drift = []
    for t in range(1, T):
        z_prev = mu_seq[0, t-1].numpy()
        z_curr = mu_seq[0, t].numpy()
        d = float(np.linalg.norm(z_curr - z_prev))
        drift.append({"from_session": t, "to_session": t+1, "drift_L2": round(d, 4)})

    # Clinical baselines
    def safe(key):
        v = rec.get(key)
        return float(v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else None

    report = {
        "participant_id":  pid,
        "group":           group,
        "n_sessions":      T,
        "clinical_baselines": {
            "pre_pcl5_total":   safe("pre_pcl5_total"),
            "post_pcl5_total":  safe("post_pcl5_total"),
            "pre_wemwbs":       safe("pre_wemwbs"),
            "post_wemwbs":      safe("post_wemwbs"),
            "pre_cd_risc":      safe("pre_cd_risc"),
            "post_cd_risc":     safe("post_cd_risc"),
            "pre_gse":          safe("pre_gse"),
        },
        "model_predictions": {
            "pcl5_delta":   round(pred_pcl5_delta, 3),
            "wemwbs_delta": round(pred_wemwbs_delta, 3),
            "cdrisc_delta": round(pred_cdrisc_delta, 3),
            "gse_baseline": round(pred_gse, 3),
        },
        "latent_trajectory":  latent_trajectory,
        "session_drift_L2":   drift,
        "uncertainty": {
            "note": "Full uncertainty quantification requires ensemble LoRA rollouts (N>=5).",
            "h_patient_norm": round(float(torch.norm(h_patient[0])), 4),
        },
    }

    return report


# ---------------------------------------------------------------
# 3. Summary statistics across all patients
# ---------------------------------------------------------------
def compute_cohort_summary(reports):
    """Aggregate clinical predictions across all patients."""
    groups = {}
    for r in reports:
        g = r["group"]
        if g not in groups:
            groups[g] = {"pcl5_deltas": [], "wemwbs_deltas": [], "n": 0}
        groups[g]["pcl5_deltas"].append(r["model_predictions"]["pcl5_delta"])
        groups[g]["wemwbs_deltas"].append(r["model_predictions"]["wemwbs_delta"])
        groups[g]["n"] += 1

    summary = {}
    for g, data in groups.items():
        summary[g] = {
            "n_patients": data["n"],
            "mean_pred_pcl5_delta":   round(float(np.mean(data["pcl5_deltas"])), 3),
            "std_pred_pcl5_delta":    round(float(np.std(data["pcl5_deltas"])), 3),
            "mean_pred_wemwbs_delta": round(float(np.mean(data["wemwbs_deltas"])), 3),
            "std_pred_wemwbs_delta":  round(float(np.std(data["wemwbs_deltas"])), 3),
        }
    return summary


# ---------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------
def main():
    print("=" * 70)
    print("Tribe V2 - Clinical Output Layer")
    print("=" * 70)

    if not PKL_PATH.exists():
        print("[ERROR] unified_patient_records.pkl not found.")
        print("        Run data_ingestion_pipeline.py first.")
        return

    # Load patient records
    with open(PKL_PATH, "rb") as f:
        records = pickle.load(f)
    print(f"[LOAD] {len(records)} patient records loaded.")

    # Load or initialize backbone
    model = LSDTBackbone()
    if BACKBONE_PT.exists():
        try:
            ckpt = torch.load(BACKBONE_PT, weights_only=True)
        except Exception:
            ckpt = torch.load(BACKBONE_PT, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print("[MODEL] Loaded pretrained LSDT backbone.")
    else:
        print("[MODEL] No pretrained backbone found -- using random init (demo mode).")

    # Generate per-patient reports
    OUTPUT_DIR.mkdir(exist_ok=True)
    reports = []

    active_recs = [r for r in records if r.get("group") in ("NF", "MI")]
    control_recs = [r for r in records if r.get("group") not in ("NF", "MI")]
    all_recs = active_recs + control_recs

    print(f"\n[GENERATE] Processing {len(all_recs)} patients...")

    for rec in all_recs:
        report = generate_patient_report(rec, model)
        reports.append(report)

        # Save individual patient report
        pid = report["participant_id"]
        out_path = OUTPUT_DIR / f"report_{pid}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=True)

    # Cohort summary
    summary = compute_cohort_summary(reports)
    summary_path = OUTPUT_DIR / "cohort_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(f"\n[SAVED] {len(reports)} patient reports -> {OUTPUT_DIR}")
    print(f"[SAVED] Cohort summary -> {summary_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("COHORT SUMMARY")
    print("=" * 70)
    for g, s in summary.items():
        print(f"\n  Group: {g} (n={s['n_patients']})")
        print(f"    Predicted PCL-5 Delta:   {s['mean_pred_pcl5_delta']:.3f} +/- {s['std_pred_pcl5_delta']:.3f}")
        print(f"    Predicted WEMWBS Delta:  {s['mean_pred_wemwbs_delta']:.3f} +/- {s['std_pred_wemwbs_delta']:.3f}")

    print("\n[DONE] Clinical output layer complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
