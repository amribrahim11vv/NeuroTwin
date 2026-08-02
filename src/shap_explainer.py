"""
shap_explainer.py
==================
Phase 3  -  SHAP Feature Attribution.

Loads the trained CQL Q-network and computes SHAP values
for the max-Q action prediction.

Generates:
  - shap_summary.txt   : ranked feature importances
  - shap_bar_plot.png  : bar chart of mean |SHAP|

Usage:
    python shap_explainer.py
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).parent.parent
PKL_PATH   = BASE_DIR / "outputs" / "unified_patient_records.pkl"
Q_NET_PATH = BASE_DIR / "models" / "cql_q_network.pt"

STATE_DIM = 35
N_ACTIONS = 3

# Feature name map  -  these are positional labels for the state vector
# Index matches session_state_vec construction in data_ingestion_pipeline.py
# NF: [allch*10, onech*10, r1_trial, g0_trial, mood_pre, stress_pre, pcl5_pre, wemwbs_pre, cdrisc_pre]
# MI: [theta, alpha, ratio, da_1s, da_2s, mood_pre, stress_pre, pcl5_pre, wemwbs_pre, cdrisc_pre]
FEATURE_NAMES_NF = (
    [f"allch_g{i+1}" for i in range(10)] +
    [f"onech_g{i+1}" for i in range(10)] +
    ["r1_trial_sec", "g0_trial_sec",
     "mood_pre", "stress_pre", "pcl5_pre", "wemwbs_pre", "cdrisc_pre"] +
    [f"pad_{i}" for i in range(35 - 27)]
)
FEATURE_NAMES_MI = (
    ["theta_class1", "alpha_class2", "theta_alpha_ratio",
     "da_1sec_acc",  "da_2sec_acc",
     "mood_pre", "stress_pre", "pcl5_pre", "wemwbs_pre", "cdrisc_pre"] +
    [f"pad_{i}" for i in range(35 - 10)]
)
FEATURE_NAMES_GENERIC = [f"state_{i}" for i in range(STATE_DIM)]


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


def main():
    print("=" * 70)
    print("Tribe V2  -  SHAP Feature Attribution")
    print("=" * 70)

    if not PKL_PATH.exists():
        print("[ERROR] unified_patient_records.pkl not found.")
        return
    if not Q_NET_PATH.exists():
        print("[ERROR] cql_q_network.pt not found. Run cql_offline_rl.py first.")
        return

    with open(PKL_PATH, "rb") as f:
        records = pickle.load(f)

    try:
        ckpt = torch.load(Q_NET_PATH, weights_only=True)
    except Exception:
        ckpt = torch.load(Q_NET_PATH, weights_only=True)
    q_net = QNetwork()
    q_net.load_state_dict(ckpt["model_state_dict"])
    q_net.eval()
    print(f"[LOAD] Q-network loaded")

    # Build data matrix
    X, groups = [], []
    for rec in records:
        sessions = rec.get("sessions", [])
        if not sessions:
            continue
        vec = sessions[0].get("session_state_vec", [])
        X.append(clean_vec(vec))
        groups.append(rec.get("group", "CONTROL"))

    X = np.array(X, dtype=np.float32)
    print(f"[SHAP] Data matrix: {X.shape}")

    # Predict function: returns max Q value
    def predict_max_q(x_np):
        x_t = torch.FloatTensor(x_np)
        with torch.no_grad():
            return q_net(x_t).numpy().max(axis=1)

    # KernelSHAP with small background (all patients as background)
    n_bg = min(len(X), 10)
    explainer = shap.KernelExplainer(predict_max_q, X[:n_bg])

    print(f"[SHAP] Computing SHAP values for {len(X)} patients (100 samples)...")
    shap_vals = explainer.shap_values(X, nsamples=100, silent=True)
    # shap_vals: [N, STATE_DIM]

    mean_abs = np.abs(shap_vals).mean(axis=0)

    # Determine feature names (use NF names as default since most patients are NF/MI mix)
    feature_names = FEATURE_NAMES_GENERIC[:STATE_DIM]

    # Rank features
    ranked = sorted(enumerate(mean_abs), key=lambda x: -x[1])

    print("\n[SHAP] Feature Importance Ranking:")
    print(f"  {'Rank':<5} {'Feature':<30} {'Mean |SHAP|'}")
    print("  " + "-" * 50)
    for rank, (feat_idx, importance) in enumerate(ranked[:20], 1):
        fname = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feat_{feat_idx}"
        print(f"  {rank:<5} {fname:<30} {importance:.6f}")

    # Save summary
    summary_path = BASE_DIR / "outputs" / "shap_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Tribe V2  -  SHAP Feature Attribution Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"N patients: {len(X)}\n\n")
        f.write(f"{'Rank':<5} {'Feature':<30} {'Mean |SHAP|'}\n")
        f.write("-" * 50 + "\n")
        for rank, (feat_idx, importance) in enumerate(ranked, 1):
            fname = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feat_{feat_idx}"
            f.write(f"{rank:<5} {fname:<30} {importance:.6f}\n")

    # Bar plot  -  top 15 features
    plot_path = BASE_DIR / "outputs" / "shap_bar_plot.png"
    top_n = min(15, len(ranked))
    top_indices     = [ranked[i][0] for i in range(top_n)]
    top_importances = [ranked[i][1] for i in range(top_n)]
    top_names       = [feature_names[fi] if fi < len(feature_names) else f"feat_{fi}"
                       for fi in top_indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#4e79a7" if "pcl5" in n or "mood" in n or "stress" in n else
              "#f28e2b" if "theta" in n or "alpha" in n or "allch" in n else
              "#59a14f" for n in top_names]
    bars = ax.barh(top_names[::-1], top_importances[::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title("Top Feature Importances  -  CQL Q-Network\n(Tribe V2 PTSD Digital Brain Twin)", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\n[SAVED] Summary -> {summary_path}")
    print(f"[SAVED] Bar plot -> {plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
