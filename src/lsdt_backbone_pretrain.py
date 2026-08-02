"""
lsdt_backbone_pretrain.py
==========================
Phase 2 - LSDT Backbone Pretraining.

Trains a shared beta-VAE + GRU backbone on pooled NF+MI session data.
Architecture:
  - Encoder:  MLP [D -> 128 -> 64 -> 2*Z]  (Z=16)
  - Decoder:  MLP [Z -> 64 -> 128 -> D]
  - GRU:      [Z -> 32 hidden]  (1 layer)
  - beta-VAE:    ELBO with beta annealed 0->1 over 50 epochs
  - Auxiliary heads on h_patient for PCL-5Delta, WEMWBS Delta, CD-RISC Delta

Usage:
    python lsdt_backbone_pretrain.py
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).parent.parent
PKL_PATH   = BASE_DIR / "outputs" / "unified_patient_records.pkl"
MODEL_PATH = BASE_DIR / "models" / "lsdt_backbone.pt"

# -- Hyperparameters ------------------------------------------
LATENT_DIM  = 16
GRU_HIDDEN  = 32
INPUT_DIM   = 35   # max state vector dimension
EPOCHS      = 200
LR          = 1e-3
BETA_WARMUP = 50   # epochs to anneal beta from 0->1
LAMBDA_AUX  = 0.5  # auxiliary head loss weight (decays to 0.1 at epoch 30)
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# -- Dataset --------------------------------------------------
class PatientSessionDataset(Dataset):
    """
    Each sample = one patient's full session sequence.
    Returns:
      x_seq   : [T, INPUT_DIM]  padded/masked session state vectors
      mask    : [T]             boolean - True = real session
      clinical: [4]             [pcl5_delta, wemwbs_delta, cd_risc_delta, gse_pre]
      group   : str
      pid     : str
    """
    def __init__(self, records: list[dict], max_sessions: int = 10):
        self.records     = records
        self.max_sessions = max_sessions

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec  = self.records[idx]
        sessions = rec.get("sessions", [])
        T = min(len(sessions), self.max_sessions)

        x_seq = np.zeros((self.max_sessions, INPUT_DIM), dtype=np.float32)
        mask  = np.zeros(self.max_sessions, dtype=bool)

        for t, sess in enumerate(sessions[:T]):
            vec = sess.get("session_state_vec", [])
            vec = [v if (v is not None and not (isinstance(v, float) and np.isnan(v))) else 0.0
                   for v in vec]
            vec = vec[:INPUT_DIM]
            vec += [0.0] * (INPUT_DIM - len(vec))
            x_seq[t] = np.array(vec, dtype=np.float32)
            mask[t]  = True

        # Clinical targets
        def safe(key):
            v = rec.get(key)
            return float(v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else 0.0

        pre_pcl5  = safe("pre_pcl5_total")
        post_pcl5 = safe("post_pcl5_total")
        pre_wemwbs  = safe("pre_wemwbs");  post_wemwbs  = safe("post_wemwbs")
        pre_cdrisc  = safe("pre_cd_risc"); post_cdrisc  = safe("post_cd_risc")

        clinical = np.array([
            post_pcl5  - pre_pcl5,    # pcl5_delta
            post_wemwbs - pre_wemwbs,  # wemwbs_delta
            post_cdrisc - pre_cdrisc,  # cd_risc_delta
            safe("pre_gse"),           # gse_baseline
        ], dtype=np.float32)

        return (
            torch.FloatTensor(x_seq),
            torch.BoolTensor(mask),
            torch.FloatTensor(clinical),
            rec.get("group", "UNKNOWN"),
            rec.get("participant_id", "?"),
        )


# -- Model ----------------------------------------------------
class LSDTBackbone(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM,
                 latent_dim: int = LATENT_DIM,
                 gru_hidden: int = GRU_HIDDEN):
        super().__init__()
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.gru_hidden = gru_hidden

        # VAE Encoder: D -> 128 -> 64 -> 2Z
        self.vae_encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64),        nn.LayerNorm(64),  nn.ReLU(),
            nn.Linear(64, 2 * latent_dim),
        )

        # VAE Decoder: Z -> 64 -> 128 -> D
        self.vae_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.LayerNorm(64),  nn.ReLU(),
            nn.Linear(64, 128),         nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, input_dim),
        )

        # GRU temporal integrator: Z -> 32
        self.gru = nn.GRU(latent_dim, gru_hidden, batch_first=True)

        # Auxiliary regression heads on h_patient in R^32
        self.head_pcl5   = nn.Linear(gru_hidden, 1)
        self.head_wemwbs = nn.Linear(gru_hidden, 1)
        self.head_cdrisc = nn.Linear(gru_hidden, 1)
        self.head_gse    = nn.Linear(gru_hidden, 1)

    def encode(self, x: torch.Tensor):
        """x: [B, D] -> mu, log_var: [B, Z]"""
        out = self.vae_encoder(x)
        mu, log_var = out.chunk(2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        return self.vae_decoder(z)

    def forward(self, x_seq: torch.Tensor, mask: torch.Tensor):
        """
        x_seq: [B, T, D]
        mask:  [B, T]  True = real session
        Returns: recon, mu_list, lv_list, h_patient, aux_preds
        """
        B, T, D = x_seq.shape
        mu_list, lv_list, z_list, recon_list = [], [], [], []

        for t in range(T):
            xt = x_seq[:, t, :]      # [B, D]
            mu_t, lv_t = self.encode(xt)
            z_t = self.reparameterize(mu_t, lv_t)
            recon_t = self.decode(z_t)
            mu_list.append(mu_t)
            lv_list.append(lv_t)
            z_list.append(z_t)
            recon_list.append(recon_t)

        z_seq    = torch.stack(z_list, dim=1)     # [B, T, Z]
        recon    = torch.stack(recon_list, dim=1)  # [B, T, D]
        mu_all   = torch.stack(mu_list, dim=1)     # [B, T, Z]
        lv_all   = torch.stack(lv_list, dim=1)

        # GRU over latent sequence - use mask to find last real timestep
        h_seq, _ = self.gru(z_seq)   # [B, T, gru_hidden]

        # Patient summary = last real session's hidden state
        last_idx = mask.long().sum(dim=1).clamp(min=1) - 1  # [B]
        h_patient = h_seq[torch.arange(B), last_idx, :]      # [B, gru_hidden]

        # Auxiliary predictions
        aux_preds = torch.stack([
            self.head_pcl5(h_patient).squeeze(-1),
            self.head_wemwbs(h_patient).squeeze(-1),
            self.head_cdrisc(h_patient).squeeze(-1),
            self.head_gse(h_patient).squeeze(-1),
        ], dim=-1)   # [B, 4]

        return recon, mu_all, lv_all, h_patient, aux_preds


# -- Loss -----------------------------------------------------
def compute_elbo_loss(recon, x_seq, mu, log_var, mask, beta):
    """
    recon, x_seq: [B, T, D]
    mu, log_var:  [B, T, Z]
    mask:         [B, T]
    """
    mask_exp = mask.unsqueeze(-1).float()  # [B, T, 1]

    # Reconstruction: MSE over real sessions only
    recon_loss = F.mse_loss(recon * mask_exp, x_seq * mask_exp, reduction="sum")
    n_valid = mask.float().sum().clamp(min=1)
    recon_loss = recon_loss / n_valid

    # KL divergence
    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    kl = (kl * mask_exp).sum() / n_valid

    return recon_loss + beta * kl, recon_loss, kl


# -- Training loop --------------------------------------------
def train(records: list[dict]):
    # Filter to NF + MI only (Control has no EEG sessions)
    train_recs = [r for r in records if r.get("group") in ("NF", "MI")]
    print(f"[TRAIN] NF+MI participants for backbone: {len(train_recs)}")

    dataset    = PatientSessionDataset(train_recs)
    loader     = DataLoader(dataset, batch_size=4, shuffle=True,
                            collate_fn=lambda b: b)

    model     = LSDTBackbone()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        beta    = min(1.0, epoch / BETA_WARMUP)
        lam_aux = 0.5 if epoch <= 30 else 0.1
        total_loss = 0.0
        n_batches  = 0

        for batch in loader:
            x_seqs   = torch.stack([b[0] for b in batch])  # [B, T, D]
            masks    = torch.stack([b[1] for b in batch])  # [B, T]
            clinicals = torch.stack([b[2] for b in batch]) # [B, 4]

            optimizer.zero_grad()
            recon, mu, lv, h_patient, aux_preds = model(x_seqs, masks)

            elbo, recon_loss, kl = compute_elbo_loss(recon, x_seqs, mu, lv, masks, beta)

            # Auxiliary loss on clinical targets (skip zero-padded targets)
            aux_loss = F.mse_loss(aux_preds, clinicals)
            loss = elbo + lam_aux * aux_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = deepcopy(model.state_dict())

        if epoch % 25 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | beta={beta:.2f} | "
                  f"loss={avg_loss:.4f} | recon={recon_loss:.4f} | kl={kl:.4f}")

    model.load_state_dict(best_state)
    return model


# -- Main -----------------------------------------------------
def main():
    print("=" * 70)
    print("Tribe V2 - LSDT Backbone Pretraining (beta-VAE + GRU)")
    print("=" * 70)

    if not PKL_PATH.exists():
        print("[ERROR] unified_patient_records.pkl not found.")
        print("        Run data_ingestion_pipeline.py first.")
        return

    with open(PKL_PATH, "rb") as f:
        records = pickle.load(f)

    print(f"[LOAD] {len(records)} patient records loaded.")

    model = train(records)

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim":  INPUT_DIM,
        "latent_dim": LATENT_DIM,
        "gru_hidden": GRU_HIDDEN,
    }, MODEL_PATH)

    print(f"\n[SAVED] Backbone saved to: {MODEL_PATH}")
    print("=" * 70)
    print("Next step: run  lora_patient_adaptation.py")


if __name__ == "__main__":
    main()
