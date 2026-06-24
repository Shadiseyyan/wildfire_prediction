import os
import math
import numpy as np

import matplotlib
matplotlib.use('Agg')   # file-only backend; change to 'TkAgg' for live popups
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import pennylane as qml


# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

class Config:
    # Spatial / temporal  (overridden from data in build_config_from_data)
    T            = 8
    H            = 64
    W            = 64
    C_in         = 10
    T_out        = 4 

    # Patch sizes  (auto-corrected in build_config_from_data)
    pt           = 2
    ph           = 8
    pw           = 8

    # Transformer
    d_model      = 256
    n_heads      = 8
    n_enc_layers = 4
    n_dec_layers = 4
    ffn_mult     = 4
    dropout      = 0.1

    # Local-window cuboid attention
    local_t      = 2
    local_h      = 4
    local_w      = 4

    # Quantum
    n_qubits     = 8
    q_depth      = 4
    q_in_dim     = 32       # must be >= n_qubits


# ══════════════════════════════════════════════════════════════════════
# 1.  3-D PATCH EMBEDDING
# ══════════════════════════════════════════════════════════════════════

class PatchEmbedding3D(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.Tt  = cfg.T // cfg.pt
        self.Th  = cfg.H // cfg.ph
        self.Tw  = cfg.W // cfg.pw

        patch_dim      = cfg.C_in * cfg.pt * cfg.ph * cfg.pw
        self.proj      = nn.Linear(patch_dim, cfg.d_model)
        self.norm      = nn.LayerNorm(cfg.d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.Tt * self.Th * self.Tw, cfg.d_model) * 0.02
        )

    def forward(self, x):
        cfg  = self.cfg
        x    = rearrange(
            x,
            'b (tt pt) (th ph) (tw pw) c -> b tt th tw (pt ph pw c)',
            pt=cfg.pt, ph=cfg.ph, pw=cfg.pw,
        )                                              
        x    = self.norm(self.proj(x))                 
        flat = rearrange(x, 'b tt th tw d -> b (tt th tw) d')
        flat = flat + self.pos_embed
        x    = rearrange(
            flat, 'b (tt th tw) d -> b tt th tw d',
            tt=self.Tt, th=self.Th, tw=self.Tw,
        )
        return x                                       


# ══════════════════════════════════════════════════════════════════════
# 2.  CUBOID ATTENTION  
# ══════════════════════════════════════════════════════════════════════

class CuboidAttention(nn.Module):
    def __init__(self, d_model, n_heads, local=True,
                 local_t=2, local_h=4, local_w=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.local    = local
        self.lt, self.lh, self.lw = local_t, local_h, local_w
        self.scale    = self.head_dim ** -0.5

        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, context=None):
        B, Tt, Th, Tw, D = x.shape
        is_cross = context is not None
        kv_src   = context if is_cross else x

        if self.local and not is_cross:
            lt = min(self.lt, Tt)
            lh = min(self.lh, Th)
            lw = min(self.lw, Tw)
            x_win = rearrange(
                x,
                'b (tt lt) (th lh) (tw lw) d -> (b tt th tw) (lt lh lw) d',
                lt=lt, lh=lh, lw=lw
            )
            qkv = self.qkv(x_win)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            q_flat = rearrange(x,      'b tt th tw d -> b (tt th tw) d')
            k_flat = rearrange(kv_src, 'b tt th tw d -> b (tt th tw) d')
            
            q = self.qkv(q_flat)[..., :D]
            kv_in = self.qkv(k_flat)
            k, v  = kv_in[..., D:2*D], kv_in[..., 2*D:]

        def split_heads(t):
            return rearrange(t, '... n (h d) -> ... h n d', h=self.n_heads)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        
        attn = torch.einsum('...hqd,...hkd->...hqk', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        
        out  = torch.einsum('...hqk,...hkd->...hqd', attn, v)
        out  = rearrange(out, '... h n d -> ... n (h d)')
        out  = self.proj(out)

        if self.local and not is_cross:
            lt = min(self.lt, Tt); lh = min(self.lh, Th); lw = min(self.lw, Tw)
            ntt, nth, ntw = Tt//lt, Th//lh, Tw//lw
            out = rearrange(out, '(b tt th tw) (lt lh lw) d -> b (tt lt) (th lh) (tw lw) d',
                            b=B, tt=ntt, th=nth, tw=ntw, lt=lt, lh=lh, lw=lw)
        else:
            out = rearrange(out, 'b (tt th tw) d -> b tt th tw d',
                            b=B, tt=Tt, th=Th, tw=Tw)
        return out


# ══════════════════════════════════════════════════════════════════════
# 3.  ENCODER BLOCK
# ══════════════════════════════════════════════════════════════════════
 
class EncoderBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.local_attn  = CuboidAttention(
            d, cfg.n_heads, local=True,
            local_t=cfg.local_t, local_h=cfg.local_h, local_w=cfg.local_w,
            dropout=cfg.dropout,
        )
        self.global_attn = CuboidAttention(
            d, cfg.n_heads, local=False, dropout=cfg.dropout,
        )
        self.merge = nn.Linear(2 * d, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn   = nn.Sequential(
            nn.Linear(d, d * cfg.ffn_mult),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d * cfg.ffn_mult, d),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        merged = self.merge(
            torch.cat([self.local_attn(x), self.global_attn(x)], dim=-1)
        )
        x = self.norm1(x + merged)
        x = self.norm2(x + self.ffn(x))
        return x


# ══════════════════════════════════════════════════════════════════════
# 4.  DECODER BLOCK
# ══════════════════════════════════════════════════════════════════════

class DecoderBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.self_attn  = CuboidAttention(
            d, cfg.n_heads, local=True,
            local_t=cfg.local_t, local_h=cfg.local_h, local_w=cfg.local_w,
            dropout=cfg.dropout,
        )
        self.cross_attn = CuboidAttention(
            d, cfg.n_heads, local=False, dropout=cfg.dropout,
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.ffn   = nn.Sequential(
            nn.Linear(d, d * cfg.ffn_mult),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d * cfg.ffn_mult, d),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, enc_mem):
        x = self.norm1(x + self.self_attn(x))
        x = self.norm2(x + self.cross_attn(x, context=enc_mem))
        x = self.norm3(x + self.ffn(x))
        return x


# ══════════════════════════════════════════════════════════════════════
# 5.  VQC DEFINITION
# ══════════════════════════════════════════════════════════════════════

def build_vqc(n_qubits: int, q_depth: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        for i in range(n_qubits):
            qml.RX(inputs[i], wires=i)
        for layer in range(q_depth):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


# ══════════════════════════════════════════════════════════════════════
# 6.  QML BRIDGE
# ══════════════════════════════════════════════════════════════════════

class QMLBridge(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg         = cfg
        self.pre_proj    = nn.Linear(cfg.d_model, cfg.q_in_dim)
        self.post_proj   = nn.Linear(cfg.n_qubits, cfg.d_model)
        self.norm        = nn.LayerNorm(cfg.d_model)
        self.vqc_weights = nn.Parameter(
            torch.randn(cfg.q_depth, cfg.n_qubits, 2) * 0.1
        )
        self.input_scale = nn.Parameter(
            torch.ones(cfg.n_qubits) * math.pi
        )
        self.circuit     = build_vqc(cfg.n_qubits, cfg.q_depth)

    def forward(self, enc_states):
        B = enc_states.shape[0]
        pooled = enc_states.mean(dim=(1, 2, 3))

        reduced     = torch.tanh(self.pre_proj(pooled))                    
        angles      = (reduced[:, :self.cfg.n_qubits]
                       * self.input_scale).float()                         
        weights_f32 = self.vqc_weights.float()                             

        q_outs = []
        for i in range(B):
            result = self.circuit(angles[i], weights_f32)
            q_outs.append(torch.stack(result).float())
        q_out = torch.stack(q_outs).float()                                

        bias = self.post_proj(q_out)            
        bias = bias[:, None, None, None, :]     

        return self.norm(enc_states + bias)     


# ══════════════════════════════════════════════════════════════════════
# 7.  PREDICTION HEADS
# ══════════════════════════════════════════════════════════════════════

class PredictionHeads(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d             = cfg.d_model
        self.pt       = cfg.pt
        self.ph       = cfg.ph
        self.pw       = cfg.pw
        out_patch_dim = cfg.pt * cfg.ph * cfg.pw   

        self.spread_head   = nn.Linear(d, out_patch_dim)
        self.prob_head     = nn.Linear(d, out_patch_dim)
        self.ignition_head = nn.Sequential(
            nn.Linear(d, out_patch_dim), nn.Sigmoid()
        )
        self.burned_head   = nn.Linear(d, out_patch_dim)

    def _to_pixels(self, patch_out):
        x = rearrange(
            patch_out,
            'b tt th tw (pt ph pw) -> b (tt pt) (th ph) (tw pw)',
            pt=self.pt, ph=self.ph, pw=self.pw,
        )
        return x.unsqueeze(2)   

    def forward(self, x):
        return {
            "spread_forecast":  self._to_pixels(self.spread_head(x)),
            "fire_probability": self._to_pixels(self.prob_head(x)),
            "ignition_risk":    self._to_pixels(self.ignition_head(x)),
            "burned_area":      F.relu(self._to_pixels(self.burned_head(x))),
        }


# ══════════════════════════════════════════════════════════════════════
# 8.  FULL MODEL
# ══════════════════════════════════════════════════════════════════════

class EarthformerQML(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg         = cfg
        self.patch_embed = PatchEmbedding3D(cfg)
        self.encoder     = nn.ModuleList(
            [EncoderBlock(cfg) for _ in range(cfg.n_enc_layers)]
        )
        self.qml_bridge  = QMLBridge(cfg)
        self.decoder     = nn.ModuleList(
            [DecoderBlock(cfg) for _ in range(cfg.n_dec_layers)]
        )
        self.heads       = PredictionHeads(cfg)

        Tt_out         = cfg.T_out // cfg.pt
        Th             = cfg.H     // cfg.ph
        Tw             = cfg.W     // cfg.pw
        self.dec_query = nn.Parameter(
            torch.randn(1, Tt_out, Th, Tw, cfg.d_model) * 0.02
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B      = x.shape[0]
        tokens = self.patch_embed(x)        

        enc = tokens
        for layer in self.encoder:
            enc = layer(enc)                

        enc_q = self.qml_bridge(enc)        

        dec = repeat(self.dec_query, '1 tt th tw d -> b tt th tw d', b=B)
        for layer in self.decoder:
            dec = layer(dec, enc_q)         

        return self.heads(dec)              


# ══════════════════════════════════════════════════════════════════════
# 9.  LOSS ALIGNMENT 
# ══════════════════════════════════════════════════════════════════════

class WildfireLoss(nn.Module):
    def __init__(self, pos_weight=10.0,
                 lam_spread=1.0, lam_prob=1.0,
                 lam_ignition=0.5, lam_burned=0.5):
        super().__init__()
        pw = torch.tensor([pos_weight])
        self.bce_spread   = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.bce_prob     = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.bce_ignition = nn.BCELoss()
        self.mse_burned   = nn.MSELoss()
        self.lam = dict(
            spread=lam_spread, prob=lam_prob,
            ignition=lam_ignition, burned=lam_burned,
        )

    def align_target(self, prd, tgt):
        """Forces the 4D target to match the 5D prediction shape."""
        # Insert Time dimension -> [B, 1, C, H, W] if missing
        if tgt.dim() == prd.dim() - 1:
            tgt = tgt.unsqueeze(1)
            
        # Duplicate the target to match predicted timesteps if needed
        if tgt.shape != prd.shape:
            tgt = tgt.expand_as(prd)
            
        return tgt.clamp(0.0, 1.0) # Prevents BCE NaN errors

    def forward(self, preds, targets):
        tgt_spread = self.align_target(preds["spread_forecast"], targets["spread_forecast"])
        tgt_prob   = self.align_target(preds["fire_probability"], targets["fire_probability"])
        tgt_ign    = self.align_target(preds["ignition_risk"], targets["ignition_risk"])
        tgt_burn   = self.align_target(preds["burned_area"], targets["burned_area"])

        loss  = self.lam["spread"]   * self.bce_spread(preds["spread_forecast"], tgt_spread)
        loss += self.lam["prob"]     * self.bce_prob(preds["fire_probability"], tgt_prob)
        loss += self.lam["ignition"] * self.bce_ignition(preds["ignition_risk"], tgt_ign)
        loss += self.lam["burned"]   * self.mse_burned(preds["burned_area"], tgt_burn)
        
        return loss


# ══════════════════════════════════════════════════════════════════════
# 10. DATASET 
# ══════════════════════════════════════════════════════════════════════
class TrainAndTarDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  
        y = self.Y[idx]  
        
        # We must move the Channel dimension from the end to the middle
        # Convert (Time, Height, Width, Channel) -> (Time, Channel, Height, Width)
        if y.dim() == 4 and y.shape[-1] == 1:
            y = y.permute(0, 3, 1, 2)
        elif y.dim() == 3 and y.shape[-1] == 1:
            y = y.permute(2, 0, 1)  
        elif y.dim() == 2:
            y = y.unsqueeze(0)      
            
        targets = {
            "spread_forecast":  y,
            "fire_probability": y,
            "ignition_risk":    y,
            "burned_area":      y,
        }
        return x, targets

# ══════════════════════════════════════════════════════════════════════
# 11. CONFIG BUILDER 
# ══════════════════════════════════════════════════════════════════════

def build_config_from_data(X: np.ndarray, Y: np.ndarray) -> Config:
    cfg          = Config()
    cfg.T        = X.shape[1]
    cfg.H        = X.shape[2]
    cfg.W        = X.shape[3]
    cfg.C_in     = X.shape[4]
    
    # Identify true sequence length safely
    if Y.ndim == 4 and Y.shape[-1] in (1, 3):
        cfg.T_out = 1 # Static image target (B, H, W, C)
    elif Y.ndim >= 4:
        cfg.T_out = Y.shape[1]
    else:
        cfg.T_out = 1

    # Lightweight settings for a manageable training run
    cfg.n_enc_layers = 2
    cfg.n_dec_layers = 2
    cfg.d_model      = 64
    cfg.n_heads      = 4
    cfg.n_qubits     = 4
    cfg.q_depth      = 2
    cfg.q_in_dim     = 8 

    # Auto-correct spatial patch sizes 
    for attr, dim in [('ph', cfg.H), ('pw', cfg.W)]:
        p = getattr(cfg, attr)
        while p > 1 and dim % p != 0:
            p -= 1
        setattr(cfg, attr, p)
        
    # Temporal patch must divide BOTH input sequences and output sequences evenly
    p_t = cfg.pt
    while p_t > 1 and (cfg.T % p_t != 0 or cfg.T_out % p_t != 0):
        p_t -= 1
    cfg.pt = p_t

    Tt_in = cfg.T // cfg.pt
    Tt_out = cfg.T_out // cfg.pt
    Th = cfg.H // cfg.ph
    Tw = cfg.W // cfg.pw

    while cfg.local_t > 1 and (Tt_in % cfg.local_t != 0 or Tt_out % cfg.local_t != 0):
        cfg.local_t -= 1
    while cfg.local_h > 1 and (Th % cfg.local_h != 0):
        cfg.local_h -= 1
    while cfg.local_w > 1 and (Tw % cfg.local_w != 0):
        cfg.local_w -= 1

    return cfg


# ══════════════════════════════════════════════════════════════════════
# 12. TRAINING
# ══════════════════════════════════════════════════════════════════════

def train_on_trainandtar(
    train_dir:  str   = "trainandtars",
    n_epochs:   int   = 10,
    batch_size: int   = 4,
    lr:         float = 3e-4,
):
    X = np.load(os.path.join(train_dir, "test_input.npy"))
    Y = np.load(os.path.join(train_dir, "test_target.npy"))
    print(f"Loaded  X:{X.shape}  Y:{Y.shape}")

    cfg       = build_config_from_data(X, Y)
    dataset   = TrainAndTarDataset(X, Y)
    loader    = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
    )

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = EarthformerQML(cfg).to(device)
    criterion = WildfireLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs,
    )

    print(f"Training {len(dataset)} samples | batch {batch_size} | device {device}")
    for epoch in range(1, n_epochs + 1):
        model.train()
        running = 0.0
        for x, targets in loader:
            x       = x.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            optimizer.zero_grad()
            loss = criterion(model(x), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running += loss.item() * x.size(0)
            
        scheduler.step()
        print(f"  Epoch {epoch:>3}/{n_epochs}  loss={running / len(dataset):.6f}")

    return model, cfg


# ══════════════════════════════════════════════════════════════════════
# 13. EVALUATION -- MULTI-THRESHOLD CSI
# ══════════════════════════════════════════════════════════════════════

def calculate_multi_threshold_csi(
    model,
    dataloader,
    device,
    thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
):
    model.eval()
    stats = {t: {'tp': 0.0, 'fp': 0.0, 'fn': 0.0} for t in thresholds}

    with torch.no_grad():
        for x, targets in dataloader:
            x      = x.to(device)
            y_true = targets["fire_probability"].to(device)
            probs  = torch.sigmoid(model(x)["fire_probability"])
            
            # Align shape for evaluation safely
            if y_true.dim() == probs.dim() - 1:
                y_true = y_true.unsqueeze(1)
            if y_true.shape != probs.shape:
                y_true = y_true.expand_as(probs)

            for t in thresholds:
                pb = (probs  > t  ).float()
                lb = (y_true > 0.5).float()
                stats[t]['tp'] += (pb * lb       ).sum().item()
                stats[t]['fp'] += (pb * (1 - lb) ).sum().item()
                stats[t]['fn'] += ((1 - pb) * lb ).sum().item()

    return {
        t: s['tp'] / max(s['tp'] + s['fp'] + s['fn'], 1e-9)
        for t, s in stats.items()
    }


# ══════════════════════════════════════════════════════════════════════
# 14. EVALUATION -- PROBABILITY HISTOGRAM
# ══════════════════════════════════════════════════════════════════════

def plot_prediction_histogram(model, dataloader, device):
    model.eval()
    all_probs = []
    print("Collecting predictions for histogram ...")
    with torch.no_grad():
        for x, _ in dataloader:
            probs = torch.sigmoid(model(x.to(device))["fire_probability"])
            all_probs.extend(probs.cpu().numpy().flatten())

    all_probs = np.array(all_probs)
    fig, ax   = plt.subplots(figsize=(10, 6))
    ax.hist(all_probs, bins=100, color='crimson', alpha=0.7, log=True)
    ax.axvline(
        all_probs.mean(), color='blue', linestyle='--', linewidth=1,
        label=f'Mean: {all_probs.mean():.4f}',
    )
    ax.set_title("Distribution of Wildfire Risk Predictions (Log Scale)")
    ax.set_xlabel("Predicted Probability  (0 = Low Risk, 1 = High Risk)")
    ax.set_ylabel("Number of Pixels (log count)")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.savefig("prediction_histogram.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(
        f"Saved prediction_histogram.png  "
        f"(max={all_probs.max():.4f}, mean={all_probs.mean():.4f})"
    )


# ══════════════════════════════════════════════════════════════════════
# 15. EVALUATION -- SPATIAL HEATMAP (STATIC)
# ══════════════════════════════════════════════════════════════════════

def visualize_prediction(x_batch, y_true, y_pred_prob, batch_idx=0, time_idx=0):
    try:
        import seaborn as sns
    except ImportError:
        import subprocess, sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "seaborn", "-q"]
        )
        import seaborn as sns
        
    if y_true.dim() == 4:
        y_true = y_true.unsqueeze(1)
        
    t_idx = min(time_idx, y_true.shape[1] - 1)

    img_true = y_true[batch_idx, t_idx, 0].cpu().numpy()
    img_prob = y_pred_prob[batch_idx, time_idx, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(img_true, ax=axes[0], cmap="YlOrRd", cbar=True)
    axes[0].set_title("Actual Fire Risk")
    sns.heatmap(img_prob, ax=axes[1], cmap="YlOrRd", cbar=True)
    axes[1].set_title("AI Predicted Probability")

    save_path = "prediction_heatmap.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {os.path.abspath(save_path)}")


 # ══════════════════════════════════════════════════════════════════════
# 15.5 EVALUATION -- GEOSPATIAL MAP & COORDINATE EXTRACTION
# ══════════════════════════════════════════════════════════════════════

def get_predicted_coordinates(grid, lat_bounds, lon_bounds):
    """
    Translates the highest-probability pixel index into real Lat/Lon coordinates.
    """
    idx = np.unravel_index(np.argmax(grid, axis=None), grid.shape)
    max_val = grid[idx]

    h, w = grid.shape
    lat_range = lat_bounds[1] - lat_bounds[0]
    lon_range = lon_bounds[1] - lon_bounds[0]
    
    # Grid [0,0] is top-left (Max Lat, Min Lon)
    pred_lat = lat_bounds[1] - (idx[0] * (lat_range / h))
    pred_lon = lon_bounds[0] + (idx[1] * (lon_range / w))
    
    return (pred_lat, pred_lon), max_val

def visualize_spatial_heatmap(y_pred_prob, lat_bounds, lon_bounds, batch_idx=0, time_idx=0, save_name="predicted_fire_map.html"):
    try:
        import folium
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "folium", "-q"])
        import folium
        
    import matplotlib as mpl

    # Extract grid and coordinates
    img_prob = y_pred_prob[batch_idx, time_idx, 0].cpu().numpy()
    (p_lat, p_lon), p_val = get_predicted_coordinates(img_prob, lat_bounds, lon_bounds)
    
    print(f"\n📍 AI PREDICTED FIRE LOCATION:")
    print(f"   Latitude:  {p_lat:.5f}")
    print(f"   Longitude: {p_lon:.5f}")
    print(f"   Confidence: {p_val:.2%}")

    # Build Map
    m = folium.Map(location=[p_lat, p_lon], zoom_start=10, tiles='CartoDB dark_matter')

    # Add heatmap overlay
    norm_img = np.clip(img_prob, 0, 1)
    cmap = mpl.colormaps['YlOrRd']
    colored_img = cmap(norm_img)

    folium.raster_layers.ImageOverlay(
        image=colored_img,
        bounds=[[lat_bounds[0], lon_bounds[0]], [lat_bounds[1], lon_bounds[1]]],
        opacity=0.6,
        name="Fire Risk Grid",
    ).add_to(m)

    # Add Marker at exact coordinates
    folium.Marker(
        location=[p_lat, p_lon],
        popup=f"AI Predicted Ignition\nRisk: {p_val:.2%}",
        icon=folium.Icon(color='red', icon='fire') 
    ).add_to(m)

    m.save(save_name)
    print(f"Saved interactive map with coordinates to {os.path.abspath(save_name)}")
# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print(" Earthformer + QML  |  Wildfire Prediction (Inference)")
    print("=" * 55)

    # ── 1. Skip Training & Load Weights ───────────────────────────────
    print("\nSkipping training, loading pre-trained model ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_dummy = np.load(os.path.join(".", "train_input.npy"))
    Y_dummy = np.load(os.path.join(".", "train_target.npy"))
    cfg = build_config_from_data(X_dummy, Y_dummy)
    
    model = EarthformerQML(cfg).to(device)
    try:
        model.load_state_dict(torch.load("earthformer_qml_weights.pth", map_location=device, weights_only=True))
        print("Model weights successfully loaded from disk!")
    except FileNotFoundError:
        print("WARNING: 'earthformer_qml_weights.pth' not found. Using untrained model.")

    # ── 2. Run Predictions ────────────────────────────────────────────
    eval_ds = TrainAndTarDataset(X_dummy, Y_dummy)
    eval_dl = torch.utils.data.DataLoader(eval_ds, batch_size=4, shuffle=False)

    print("\nGenerating predictions ...")
    model.eval()
    with torch.no_grad():
        x_b, t_b = next(iter(eval_dl))
        x_b      = x_b.to(device)
        probs    = torch.sigmoid(model(x_b)["fire_probability"])

    # ── 3. Generate Visuals & Coordinates ─────────────────────────────
    
    # 3a. Static Image
    visualize_prediction(x_b, t_b["fire_probability"].to(device), probs, batch_idx=0, time_idx=0)
    
    # 3b. Interactive Map with Coordinates
    # UPDATE THESE BOUNDS BASED ON YOUR WORLD-SCALE DATASET'S CURRENT REGION
    dataset_lat_bounds = (24.3, 49.4)  # Example: Southern California Jan 2025 bounds
    dataset_lon_bounds = (-125.0, -66.9) 

    
    visualize_spatial_heatmap(
        y_pred_prob=probs, 
        lat_bounds=dataset_lat_bounds, 
        lon_bounds=dataset_lon_bounds, 
        batch_idx=0, 
        time_idx=0,
        save_name="USA2.html"
    )

    print("\nDone. Check the terminal for exact coordinates and open 'predicted_fire_map.html'.")
