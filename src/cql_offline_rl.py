"""
cql_offline_rl.py
==================
Phase 2  -  Conservative Q-Learning (CQL)  -  Offline RL.

Trains a Q-network on the 29-patient offline trajectory dataset.
MDP formulation:
  State:   session_state_vec in R^35
  Actions: 0=CONTROL, 1=MI_SESSION, 2=NF_SESSION
  Reward:  0.3*mood_delta + 0.3*(-stress_delta) + 0.4*pcl5_delta (terminal)

Algorithm: CQL (Kumar et al., 2020)  -  conservative penalty prevents
over-estimation on unseen state-action pairs.

Usage:
    python cql_offline_rl.py
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore")

BASE_DIR  = Path(__file__).parent.parent
PKL_PATH  = BASE_DIR / "outputs" / "unified_patient_records.pkl"
OUT_PATH  = BASE_DIR / "models" / "cql_q_network.pt"

# -- Hyperparameters ------------------------------------------
STATE_DIM  = 35
N_ACTIONS  = 3       # CONTROL, MI, NF
GAMMA      = 0.95
ALPHA_CQL  = 0.5    # CQL conservative penalty weight
LR         = 1e-3
EPOCHS     = 500
BATCH_SIZE = 16
SEED       = 42
# Reward weights
W_MOOD     = 0.3
W_STRESS   = 0.3
W_PCL5     = 0.4

# Action mapping
GROUP_TO_ACTION = {"CONTROL": 0, "MI": 1, "NF": 2}

torch.manual_seed(SEED)
np.random.seed(SEED)


# -- Q-Network ------------------------------------------------
class QNetwork(nn.Module):
    """MLP [35 -> 64 -> 32 -> 3]"""
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 32),        nn.LayerNorm(32), nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, s):
        return self.net(s)


# -- Trajectory extraction ------------------------------------
def extract_transitions(records: list[dict]) -> list[dict]:
    """
    Convert patient records into (s, a, r, s', done) tuples.
    """
    transitions = []

    for rec in records:
        group   = rec.get("group", "CONTROL")
        action  = GROUP_TO_ACTION.get(group, 0)
        sessions = rec.get("sessions", [])
        T = len(sessions)

        # Terminal PCL-5 improvement reward
        pcl5_delta = rec.get("pcl5_delta")
        if pcl5_delta is None or (isinstance(pcl5_delta, float) and np.isnan(pcl5_delta)):
            pcl5_delta = 0.0
        # Sign-correct: PCL-5 improvement = negative delta (post < pre)
        terminal_reward = W_PCL5 * (-pcl5_delta)

        def clean_vec(v):
            if not v:
                return np.zeros(STATE_DIM, dtype=np.float32)
            out = [x if (x is not None and not np.isnan(float(x) if isinstance(x, (int, float)) else np.nan)) else 0.0
                   for x in v]
            out = out[:STATE_DIM]
            out += [0.0] * max(0, STATE_DIM - len(out))
            return np.array(out, dtype=np.float32)

        for t_idx, sess in enumerate(sessions):
            s = clean_vec(sess.get("session_state_vec", []))
            done = (t_idx == T - 1)

            # Dense reward: mood + stress
            mood_delta   = sess.get("mood_delta", 0.0) or 0.0
            stress_delta = sess.get("stress_delta", 0.0) or 0.0
            if np.isnan(mood_delta):   mood_delta   = 0.0
            if np.isnan(stress_delta): stress_delta = 0.0

            r = W_MOOD * mood_delta + W_STRESS * (-stress_delta)
            if done:
                r += terminal_reward

            # Next state
            if not done:
                s_next = clean_vec(sessions[t_idx + 1].get("session_state_vec", []))
            else:
                s_next = np.zeros(STATE_DIM, dtype=np.float32)

            transitions.append({
                "s":      s,
                "a":      action,
                "r":      float(r),
                "s_next": s_next,
                "done":   done,
            })

    return transitions


# -- CQL Loss -------------------------------------------------
def cql_loss(q_net: QNetwork, s, a, r, s_next, done, alpha=ALPHA_CQL):
    """
    L_CQL = alpha*(log Sigma_a exp Q(s,a) - Q(s,a_data))
          + E[(r + gamma*max_a' Q(s',a') - Q(s,a))^2]
    """
    q_values = q_net(s)                         # [B, A]
    q_taken  = q_values.gather(1, a.unsqueeze(1)).squeeze(1)  # [B]

    # Conservative penalty: logsumexp over all actions - Q of taken action
    logsumexp = torch.logsumexp(q_values, dim=1)              # [B]
    conservative = (logsumexp - q_taken).mean()

    # Bellman error
    with torch.no_grad():
        q_next    = q_net(s_next)               # [B, A]
        q_next_max = q_next.max(dim=1).values   # [B]
        target    = r + GAMMA * q_next_max * (~done).float()

    bellman = F.mse_loss(q_taken, target)

    return alpha * conservative + bellman, conservative.item(), bellman.item()


# -- LOPO-CV --------------------------------------------------
def lopo_cv(records: list[dict]) -> dict:
    """Leave-One-Participant-Out Cross-Validation."""
    print("\n[LOPO-CV] Running 29 folds...")
    
    all_true_directions = []   # +1 = PCL-5 improved, -1 = worsened
    all_pred_directions = []

    for leave_out_idx, test_rec in enumerate(records):
        pid = test_rec["participant_id"]
        train_recs = [r for i, r in enumerate(records) if i != leave_out_idx]

        train_transitions = extract_transitions(train_recs)
        if not train_transitions:
            continue

        # Train a Q-network on the 28-patient dataset
        q_net = QNetwork()
        optimizer = torch.optim.Adam(q_net.parameters(), lr=LR)

        # Convert to tensors
        S      = torch.FloatTensor([t["s"]      for t in train_transitions])
        A      = torch.LongTensor( [t["a"]      for t in train_transitions])
        R      = torch.FloatTensor([t["r"]      for t in train_transitions])
        S_next = torch.FloatTensor([t["s_next"] for t in train_transitions])
        Done   = torch.BoolTensor( [t["done"]   for t in train_transitions])

        for epoch in range(200):  # Shorter epochs per fold
            q_net.train()
            idx   = np.random.choice(len(S), min(BATCH_SIZE, len(S)), replace=False)
            loss, _, _ = cql_loss(q_net,
                                  S[idx], A[idx], R[idx], S_next[idx], Done[idx])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()

        # Evaluate on test patient
        test_sessions = test_rec.get("sessions", [])
        if test_sessions:
            test_vec = test_sessions[0].get("session_state_vec", [])
            s_test = np.array(
                [x if (x is not None and not np.isnan(float(x) if isinstance(x, (int, float)) else np.nan)) else 0.0
                 for x in test_vec[:STATE_DIM]] + [0.0] * max(0, STATE_DIM - len(test_vec)),
                dtype=np.float32
            )
            q_net.eval()
            with torch.no_grad():
                q_vals = q_net(torch.FloatTensor(s_test).unsqueeze(0))
            pred_action = int(q_vals.argmax().item())
        else:
            pred_action = 0  # Default CONTROL for no-session patients

        # Ground truth: did PCL-5 improve?
        delta = test_rec.get("pcl5_delta")
        if delta is not None and not (isinstance(delta, float) and np.isnan(delta)):
            true_dir = -1 if delta < 0 else 1   # negative delta = improvement
            pred_dir = 1 if pred_action in (1, 2) else -1  # NF/MI = predicted improvement
            all_true_directions.append(true_dir)
            all_pred_directions.append(pred_dir)

        print(f"  Fold {leave_out_idx+1:02d} [{pid}]: "
              f"pred_action={['CONTROL','MI','NF'][pred_action]} | "
              f"pcl5_delta={delta:.2f}" if (delta is not None and not np.isnan(delta)) else
              f"  Fold {leave_out_idx+1:02d} [{pid}]: pred_action={['CONTROL','MI','NF'][pred_action]} | pcl5_delta=NaN")

    if all_true_directions:
        n = len(all_true_directions)
        correct = sum(1 for t, p in zip(all_true_directions, all_pred_directions) if t == p)
        accuracy = correct / n * 100
        print(f"\n[LOPO-CV] PCL-5 direction accuracy: {correct}/{n} = {accuracy:.1f}%")
        if accuracy >= 70:
            print("  [OK] MILESTONE GATE PASSED: accuracy >= 70%")
        else:
            print("  [!]  MILESTONE NOT MET: accuracy < 70%")
        return {"accuracy": accuracy, "n_folds": n}
    return {"accuracy": 0, "n_folds": 0}


# -- Full training ---------------------------------------------
def train_full(records: list[dict]) -> QNetwork:
    """Train on all 29 patients for final model."""
    transitions = extract_transitions(records)
    print(f"\n[TRAIN] Total transitions: {len(transitions)} from {len(records)} patients")

    q_net     = QNetwork()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    S      = torch.FloatTensor([t["s"]      for t in transitions])
    A      = torch.LongTensor( [t["a"]      for t in transitions])
    R      = torch.FloatTensor([t["r"]      for t in transitions])
    S_next = torch.FloatTensor([t["s_next"] for t in transitions])
    Done   = torch.BoolTensor( [t["done"]   for t in transitions])

    best_loss, best_state = float("inf"), None

    for epoch in range(1, EPOCHS + 1):
        q_net.train()
        idx = np.random.choice(len(S), min(BATCH_SIZE, len(S)), replace=False)
        loss, cons, bell = cql_loss(q_net,
                                    S[idx], A[idx], R[idx], S_next[idx], Done[idx])
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in q_net.state_dict().items()}

        if epoch % 100 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | loss={loss.item():.4f} "
                  f"| CQL={cons:.4f} | Bellman={bell:.4f}")

    q_net.load_state_dict(best_state)
    return q_net


# -- Main ------------------------------------------------------
def main():
    print("=" * 70)
    print("Tribe V2  -  Conservative Q-Learning (CQL) Offline RL")
    print("=" * 70)

    if not PKL_PATH.exists():
        print("[ERROR] unified_patient_records.pkl not found.")
        return

    with open(PKL_PATH, "rb") as f:
        records = pickle.load(f)
    print(f"[LOAD] {len(records)} patient records")

    # LOPO-CV validation
    cv_results = lopo_cv(records)

    # Full training on all patients
    print("\n[FULL TRAIN] Training on all patients...")
    q_net = train_full(records)

    # Q-value profiles for each patient
    print("\n[Q-VALUES] Per-patient intervention profiles:")
    print(f"  {'PID':<15} {'Group':<10} {'Q(Control)':<12} {'Q(MI)':<12} {'Q(NF)':<12} {'Best':<10}")
    print("  " + "-" * 65)

    for rec in records:
        sessions = rec.get("sessions", [])
        if not sessions:
            continue
        vec = sessions[0].get("session_state_vec", [])
        s = np.array(
            [x if (x is not None and not np.isnan(float(x) if isinstance(x, (int, float)) else np.nan)) else 0.0
             for x in vec[:STATE_DIM]] + [0.0] * max(0, STATE_DIM - len(vec)),
            dtype=np.float32
        )
        q_net.eval()
        with torch.no_grad():
            qv = q_net(torch.FloatTensor(s).unsqueeze(0)).squeeze().numpy()
        best = ["CONTROL", "MI", "NF"][int(np.argmax(qv))]
        print(f"  {rec['participant_id']:<15} {rec['group']:<10} "
              f"{qv[0]:<12.4f} {qv[1]:<12.4f} {qv[2]:<12.4f} {best:<10}")

    torch.save({
        "model_state_dict": q_net.state_dict(),
        "state_dim": STATE_DIM,
        "n_actions":  N_ACTIONS,
        "lopo_cv_accuracy": cv_results.get("accuracy", 0),
    }, OUT_PATH)

    print(f"\n[SAVED] Q-network saved to: {OUT_PATH}")
    print("=" * 70)
    print("Next step: run  clinical_output_layer.py")


if __name__ == "__main__":
    main()
