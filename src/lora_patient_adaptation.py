"""
lora_patient_adaptation.py
===========================
Phase 2  -  Per-Patient LoRA Adaptation using HuggingFace PEFT.

Loads the pretrained LSDT backbone and creates a separate LoRA adapter
for each of the 29 patients (rank=4, ~2K params per adapter).

Saves: lora_adapters/ directory with one adapter per participant.

Usage:
    python lora_patient_adaptation.py
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore")

BASE_DIR      = Path(__file__).parent.parent
PKL_PATH      = BASE_DIR / "outputs" / "unified_patient_records.pkl"
BACKBONE_PATH = BASE_DIR / "models" / "lsdt_backbone.pt"
ADAPTERS_DIR  = BASE_DIR / "models" / "lora_adapters"
ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

# -- Hyperparameters ------------------------------------------
LORA_RANK    = 4
LORA_ALPHA   = 8
LORA_DROPOUT = 0.1
ADAPT_EPOCHS = 100
ADAPT_LR     = 5e-4
INPUT_DIM    = 35
LATENT_DIM   = 16
GRU_HIDDEN   = 32


# -- LoRA linear layer ----------------------------------------
class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with trainable low-rank DeltaW = B*A.
    Output = W*x + (alpha/r)*B*A*x
    """
    def __init__(self, linear: nn.Linear, r: int = 4, alpha: int = 8, dropout: float = 0.1):
        super().__init__()
        self.linear  = linear
        self.r       = r
        self.scaling = alpha / r

        in_f  = linear.in_features
        out_f = linear.out_features

        self.lora_A   = nn.Linear(in_f,  r,    bias=False)
        self.lora_B   = nn.Linear(r,     out_f, bias=False)
        self.dropout  = nn.Dropout(p=dropout)

        # Initialise: A ~ N(0,1), B = 0 so DeltaW starts at 0
        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Freeze base weights
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        base   = self.linear(x)
        delta  = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base + delta


# -- Rebuild backbone -----------------------------------------
class LSDTBackbone(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM, gru_hidden=GRU_HIDDEN):
        super().__init__()
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
        self.gru         = nn.GRU(latent_dim, gru_hidden, batch_first=True)
        self.head_pcl5   = nn.Linear(gru_hidden, 1)
        self.head_wemwbs = nn.Linear(gru_hidden, 1)
        self.head_cdrisc = nn.Linear(gru_hidden, 1)
        self.head_gse    = nn.Linear(gru_hidden, 1)

    def encode(self, x):
        out = self.vae_encoder(x)
        return out.chunk(2, dim=-1)

    def reparameterize(self, mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)

    def decode(self, z):
        return self.vae_decoder(z)

    def forward(self, x_seq, mask):
        B, T, D = x_seq.shape
        z_list, recon_list, mu_list, lv_list = [], [], [], []
        for t in range(T):
            mu_t, lv_t = self.encode(x_seq[:, t])
            z_t = self.reparameterize(mu_t, lv_t)
            z_list.append(z_t); recon_list.append(self.decode(z_t))
            mu_list.append(mu_t); lv_list.append(lv_t)

        z_seq   = torch.stack(z_list, 1)
        recon   = torch.stack(recon_list, 1)
        mu_all  = torch.stack(mu_list, 1)
        lv_all  = torch.stack(lv_list, 1)
        h_seq, _ = self.gru(z_seq)
        last_idx  = mask.long().sum(1).clamp(min=1) - 1
        h_patient = h_seq[torch.arange(B), last_idx]
        aux = torch.stack([
            self.head_pcl5(h_patient).squeeze(-1),
            self.head_wemwbs(h_patient).squeeze(-1),
            self.head_cdrisc(h_patient).squeeze(-1),
            self.head_gse(h_patient).squeeze(-1),
        ], -1)
        return recon, mu_all, lv_all, h_patient, aux


class LoRABackbone(nn.Module):
    """
    Wraps LSDTBackbone, replacing the first two Linear layers of
    VAE encoder with LoRALinear modules. GRU and decoder frozen.
    """
    def __init__(self, backbone: LSDTBackbone, r: int = LORA_RANK,
                 alpha: int = LORA_ALPHA, dropout: float = LORA_DROPOUT):
        super().__init__()
        # Deep-copy so each patient gets independent params
        self.backbone = deepcopy(backbone)

        # Freeze everything first
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Inject LoRA into vae_encoder[0] and vae_encoder[3] (the Linear layers)
        enc = self.backbone.vae_encoder
        enc[0] = LoRALinear(enc[0], r=r, alpha=alpha, dropout=dropout)
        enc[3] = LoRALinear(enc[3], r=r, alpha=alpha, dropout=dropout)

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x_seq, mask):
        return self.backbone(x_seq, mask)


# -- Session -> tensors ----------------------------------------
def patient_to_tensors(rec: dict, max_sessions: int = 10):
    sessions = rec.get("sessions", [])
    T = min(len(sessions), max_sessions)
    x_seq = np.zeros((max_sessions, INPUT_DIM), dtype=np.float32)
    mask  = np.zeros(max_sessions, dtype=bool)

    for t, sess in enumerate(sessions[:T]):
        vec = sess.get("session_state_vec", [])
        vec = [v if (v is not None and not (isinstance(v, float) and np.isnan(v))) else 0.0
               for v in vec]
        vec = vec[:INPUT_DIM] + [0.0] * max(0, INPUT_DIM - len(vec))
        x_seq[t] = np.array(vec, dtype=np.float32)
        mask[t]  = True

    def safe(k):
        v = rec.get(k)
        return float(v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else 0.0

    pre_pcl5  = safe("pre_pcl5_total");  post_pcl5  = safe("post_pcl5_total")
    pre_wb    = safe("pre_wemwbs");      post_wb    = safe("post_wemwbs")
    pre_cd    = safe("pre_cd_risc");     post_cd    = safe("post_cd_risc")

    clinical = np.array([
        post_pcl5 - pre_pcl5, post_wb - pre_wb,
        post_cd   - pre_cd,   safe("pre_gse"),
    ], dtype=np.float32)

    return (
        torch.FloatTensor(x_seq).unsqueeze(0),   # [1, T, D]
        torch.BoolTensor(mask).unsqueeze(0),      # [1, T]
        torch.FloatTensor(clinical).unsqueeze(0), # [1, 4]
    )


# -- Adapt one patient -----------------------------------------
def adapt_patient(rec: dict, backbone: LSDTBackbone, epochs: int = ADAPT_EPOCHS) -> dict:
    pid   = rec["participant_id"]
    group = rec["group"]

    if group == "CONTROL":
        # Control group has no EEG sessions  -  skip LoRA adaptation
        print(f"  [{pid}] CONTROL  -  skipping LoRA adaptation (no EEG sessions)")
        return {"pid": pid, "group": group, "skipped": True}

    x_seq, mask, clinical = patient_to_tensors(rec)
    model = LoRABackbone(backbone)
    print(f"  [{pid}] Trainable params: {model.trainable_params()}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=ADAPT_LR
    )

    best_loss, best_state = float("inf"), None

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        recon, mu, lv, h_patient, aux = model(x_seq, mask)

        # ELBO
        mask_e = mask.unsqueeze(-1).float()
        recon_loss = F.mse_loss(recon * mask_e, x_seq * mask_e, reduction="sum") / mask.float().sum().clamp(1)
        kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp()) * mask_e).sum() / mask.float().sum().clamp(1)
        aux_loss = F.mse_loss(aux, clinical)
        loss = recon_loss + kl + 0.3 * aux_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = deepcopy(model.state_dict())

    # Extract only LoRA weights to save
    lora_state = {k: v for k, v in best_state.items() if "lora_" in k}
    return {
        "pid":       pid,
        "group":     group,
        "lora_state": lora_state,
        "best_loss":  best_loss,
        "skipped":    False,
    }


# -- Main ------------------------------------------------------
def main():
    print("=" * 70)
    print("Tribe V2  -  Per-Patient LoRA Adaptation")
    print("=" * 70)

    if not PKL_PATH.exists():
        print("[ERROR] unified_patient_records.pkl not found.")
        return
    if not BACKBONE_PATH.exists():
        print("[ERROR] lsdt_backbone.pt not found.")
        print("        Run lsdt_backbone_pretrain.py first.")
        return

    with open(PKL_PATH, "rb") as f:
        records = pickle.load(f)

    try:
        ckpt = torch.load(BACKBONE_PATH, weights_only=True)
    except Exception:
        ckpt = torch.load(BACKBONE_PATH, weights_only=True)
    backbone = LSDTBackbone()
    backbone.load_state_dict(ckpt["model_state_dict"])
    backbone.eval()
    print(f"[LOAD] Backbone loaded from {BACKBONE_PATH}")

    results = []
    for i, rec in enumerate(records):
        pid = rec["participant_id"]
        print(f"\n[{i+1:02d}/{len(records)}] Adapting {pid} [{rec['group']}]...")
        result = adapt_patient(rec, backbone)
        results.append(result)

        if not result.get("skipped"):
            out_path = ADAPTERS_DIR / f"{pid}_lora.pt"
            torch.save(result, out_path)
            print(f"         Saved -> {out_path.name} | best_loss={result['best_loss']:.4f}")

    print("\n" + "=" * 70)
    print(f"[DONE] Adapted {sum(1 for r in results if not r.get('skipped'))} patients")
    print(f"       Skipped {sum(1 for r in results if r.get('skipped'))} (Control)")
    print(f"       Adapters saved to: {ADAPTERS_DIR}")
    print("=" * 70)
    print("Next step: run  cql_offline_rl.py")


if __name__ == "__main__":
    main()
