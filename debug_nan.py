"""
Earthformer + QML Wildfire Prediction Model
=============================================
Architecture:
  Input sources → 3D Patch Embedding
  → Earthformer Encoder (cuboid self-attention)
  → QML Bridge (VQC via PennyLane)
  → Earthformer Decoder (cuboid cross-attention)
  → Prediction Heads (spread, probability, ignition risk, burned area)
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt  # <--- Moved to the top!

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

class Config:
    # Spatiotemporal input dims
    T           = 8          # input timesteps
    H           = 64         # spatial height (pixels)
    W           = 64         # spatial width
    C_in        = 10         # input channels (satellite bands + weather + terrain + NDVI + history)

    # Patch / cuboid
    pt          = 2          # temporal patch size
    ph          = 8          # spatial patch size (height)
    pw          = 8          # spatial patch size (width)

    # Transformer
    d_model     = 256        # hidden dim
    n_heads     = 8
    n_enc_layers = 4
    n_dec_layers = 4
    ffn_mult    = 4          # FFN hidden = d_model * ffn_mult
    dropout     = 0.1

    # Local cuboid window sizes
    local_t     = 2
    local_h     = 4
    local_w     = 4

    # QML bridge
    n_qubits    = 8          # number of qubits
    q_depth     = 4          # VQC layers (ansatz depth)
    q_in_dim    = 32         # PCA-reduced dim fed into qubit encoding (must be <= n_qubits * 2)

    # Prediction
    T_out       = 4          # output timesteps to predict
    n_classes   = 1          # binary fire / no-fire per pixel (sigmoid output)


# ─────────────────────────────────────────────
# 1. 3D PATCH EMBEDDING
# ─────────────────────────────────────────────

class PatchEmbedding3D(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.Tt = cfg.T  // cfg.pt
        self.Th = cfg.H  // cfg.ph
        self.Tw = cfg.W  // cfg.pw

        patch_dim = cfg.C_in * cfg.pt * cfg.ph * cfg.pw
        self.proj = nn.Linear(patch_dim, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)

        n_tokens = self.Tt * self.Th * self.Tw
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, cfg.d_model) * 0.02)

    def forward(self, x):
        B = x.shape[0]
        cfg = self.cfg
        x = rearrange(
            x,
            'b (tt pt) (th ph) (tw pw) c -> b tt th tw (pt ph pw c)',
            pt=cfg.pt, ph=cfg.ph, pw=cfg.pw
        )
        x = self.proj(x)
        x = self.norm(x)

        flat = rearrange(x, 'b tt th tw d -> b (tt th tw) d')
        flat = flat + self.pos_embed
        x = rearrange(flat, 'b (tt th tw) d -> b tt th tw d',
                      tt=self.Tt, th=self.Th, tw=self.Tw)
        return x


# ─────────────────────────────────────────────
# 2. CUBOID ATTENTION
# ─────────────────────────────────────────────

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
            q_flat = rearrange(x,      'b tt th tw d -> (b tt) (th tw) d')
            k_flat = rearrange(kv_src, 'b tt th tw d -> (b tt) (th tw) d')
            v_flat = k_flat
            q = self.qkv(q_flat)[..., :D]
            kv_in = self.qkv(k_flat)
            k, v  = kv_in[..., D:2*D], kv_in[..., 2*D:]
            q_flat_out = q
            q, k, v = q_flat_out, k, v

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
            out = rearrange(out, '(b tt) (th tw) d -> b tt th tw d',
                            b=B, tt=Tt, th=Th, tw=Tw)
        return out


# ─────────────────────────────────────────────
# 3. ENCODER BLOCK
# ─────────────────────────────────────────────

class EncoderBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.local_attn  = CuboidAttention(d, cfg.n_heads, local=True,
                                           local_t=cfg.local_t,
                                           local_h=cfg.local_h,
                                           local_w=cfg.local_w,
                                           dropout=cfg.dropout)
        self.global_attn = CuboidAttention(d, cfg.n_heads, local=False,
                                           dropout=cfg.dropout)
        self.merge  = nn.Linear(2 * d, d)
        self.norm1  = nn.LayerNorm(d)
        self.norm2  = nn.LayerNorm(d)
        self.ffn    = nn.Sequential(
            nn.Linear(d, d * cfg.ffn_mult),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d * cfg.ffn_mult, d),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        local_out  = self.local_attn(x)
        global_out = self.global_attn(x)
        merged = self.merge(torch.cat([local_out, global_out], dim=-1))
        x = self.norm1(x + merged)
        x = self.norm2(x + self.ffn(x))
        return x


# ─────────────────────────────────────────────
# 4. DECODER BLOCK
# ─────────────────────────────────────────────

class DecoderBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.self_attn  = CuboidAttention(d, cfg.n_heads, local=True,
                                          local_t=cfg.local_t,
                                          local_h=cfg.local_h,
                                          local_w=cfg.local_w,
                                          dropout=cfg.dropout)
        self.cross_attn = CuboidAttention(d, cfg.n_heads, local=False,
                                          dropout=cfg.dropout)
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
        x = self.norm2(x + self.ffn(x))
        return x


# ─────────────────────────────────────────────
# 5. QML BRIDGE (VARIATIONAL QUANTUM CIRCUIT)
# ─────────────────────────────────────────────

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


class QMLBridge(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg      = cfg
        self.pre_proj = nn.Linear(cfg.d_model, cfg.q_in_dim)
        self.post_proj = nn.Linear(cfg.n_qubits, cfg.d_model)
        self.norm     = nn.LayerNorm(cfg.d_model)

        self.vqc_weights = nn.Parameter(
            torch.randn(cfg.q_depth, cfg.n_qubits, 2) * 0.1
        )
        self.circuit = build_vqc(cfg.n_qubits, cfg.q_depth)
        self.input_scale = nn.Parameter(torch.ones(cfg.q_in_dim) * math.pi)

    def _run_vqc(self, x_reduced):
        n_qubits = self.cfg.n_qubits
        outputs  = []
        for sample in x_reduced:
            angles = sample[:n_qubits] * self.input_scale[:n_qubits]
            result = self.circuit(angles, self.vqc_weights)
            outputs.append(torch.stack(result))
        return torch.stack(outputs) 

    def forward(self, enc_states):
        B, Tt, Th, Tw, D = enc_states.shape
        flat = rearrange(enc_states, 'b tt th tw d -> (b tt th tw) d')

        reduced = self.pre_proj(flat)
        reduced = torch.tanh(reduced)

        q_out = self._run_vqc(reduced)
        q_out = q_out.to(reduced.dtype)

        out = self.post_proj(q_out)
        out = rearrange(out, '(b tt th tw) d -> b tt th tw d',
                        b=B, tt=Tt, th=Th, tw=Tw)
        out = self.norm(enc_states + out)
        return out


# ─────────────────────────────────────────────
# 6. PREDICTION HEADS
# ─────────────────────────────────────────────

class PredictionHeads(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.pt, self.ph, self.pw = cfg.pt, cfg.ph, cfg.pw
        
        out_patch_dim = 1 * self.pt * self.ph * self.pw

        self.spread_head   = nn.Linear(d, out_patch_dim)
        self.prob_head     = nn.Linear(d, out_patch_dim)
        self.ignition_head = nn.Sequential(nn.Linear(d, out_patch_dim), nn.Sigmoid())
        self.burned_head   = nn.Linear(d, out_patch_dim)

    def forward(self, x):
        def decode_to_pixel(patch_tensor):
            return rearrange(
                patch_tensor,
                'b tt th tw (c pt ph pw) -> b (tt pt) c (th ph) (tw pw)',
                c=1, pt=self.pt, ph=self.ph, pw=self.pw
            )

        return {
            "spread_forecast":  decode_to_pixel(self.spread_head(x)),
            "fire_probability": decode_to_pixel(self.prob_head(x)),  
            "ignition_risk":    decode_to_pixel(self.ignition_head(x)), 
            "burned_area":      F.relu(decode_to_pixel(self.burned_head(x))), 
        }

# ─────────────────────────────────────────────
# 7. FULL MODEL
# ─────────────────────────────────────────────

class EarthformerQML(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.patch_embed = PatchEmbedding3D(cfg)
        self.encoder     = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_enc_layers)])
        self.qml_bridge  = QMLBridge(cfg)
        self.decoder     = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_dec_layers)])
        self.heads       = PredictionHeads(cfg)

        Tt_out = cfg.T_out // cfg.pt
        Th     = cfg.H     // cfg.ph
        Tw     = cfg.W     // cfg.pw
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
        B = x.shape[0]
        tokens = self.patch_embed(x)
        enc = tokens
        for layer in self.encoder:
            enc = layer(enc)

        enc_q = self.qml_bridge(enc)

        dec = repeat(self.dec_query, '1 tt th tw d -> b tt th tw d', b=B)
        for layer in self.decoder:
            dec = layer(dec, enc_q)

        out = self.heads(dec)
        return out


# ─────────────────────────────────────────────
# 8. LOSS FUNCTION
# ─────────────────────────────────────────────

class WildfireLoss(nn.Module):
    def __init__(self, pos_weight=10.0, lam_spread=1.0, lam_prob=1.0,
                 lam_ignition=0.5, lam_burned=0.5):
        super().__init__()
        pw = torch.tensor([pos_weight])
        self.bce_spread   = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.bce_prob     = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.bce_ignition = nn.BCELoss()
        self.mse_burned   = nn.MSELoss()
        self.lam = dict(spread=lam_spread, prob=lam_prob,
                        ignition=lam_ignition, burned=lam_burned)

    def forward(self, preds, targets):
        loss  = self.lam["spread"]   * self.bce_spread(preds["spread_forecast"],  targets["spread_forecast"])
        loss += self.lam["prob"]     * self.bce_prob(preds["fire_probability"],   targets["fire_probability"])
        loss += self.lam["ignition"] * self.bce_ignition(preds["ignition_risk"],  targets["ignition_risk"])
        loss += self.lam["burned"]   * self.mse_burned(preds["burned_area"],      targets["burned_area"])
        return loss


# ─────────────────────────────────────────────
# 9. DATASET WRAPPER
# ─────────────────────────────────────────────

class TrainAndTarDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, cfg: Config):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.cfg = cfg

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx] 
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        y = self.Y[idx]  
        fire_2d = y.permute(2, 0, 1) 
        fire_sequence = fire_2d.unsqueeze(0).repeat(self.cfg.T_out, 1, 1, 1)

        targets = {
            "spread_forecast":  fire_sequence,
            "fire_probability": fire_sequence,
            "ignition_risk":    fire_sequence,
            "burned_area":      fire_sequence,
        }
        return x, targets


# ─────────────────────────────────────────────
# 10. TRAINING FUNCTION
# ─────────────────────────────────────────────

def train_on_trainandtar(cfg: Config, train_dir: str = "trainandtar", n_epochs: int = 10, batch_size: int = 4, lr: float = 3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on Device: {device}")

    X = np.load(os.path.join(train_dir, "train_input.npy"))
    Y = np.load(os.path.join(train_dir, "train_target.npy"))

    dataset = TrainAndTarDataset(X, Y, cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = EarthformerQML(cfg).to(device)
    criterion = WildfireLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    print(f"Starting training on {len(dataset)} samples...")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        
        for step, (x, targets) in enumerate(dataloader):
            x = x.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            optimizer.zero_grad()
            preds = model(x)
            
            loss = criterion(preds, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * x.size(0)
            
        scheduler.step()
        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch+1}/{n_epochs} | Avg Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "earthformer_qml_weights.pth")
    print("Training complete. Weights saved.")
    return model

# ─────────────────────────────────────────────
# 11. EVALUATION FUNCTION
# ─────────────────────────────────────────────

def evaluate_trainandtar(cfg: Config, model: nn.Module | None = None, train_dir: str = "trainandtar", batch_size: int = 4, threshold: float = 0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = np.load(os.path.join(train_dir, "train_input.npy"))
    Y = np.load(os.path.join(train_dir, "train_target.npy"))
    
    dataset = TrainAndTarDataset(X, Y, cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if model is None:
        model = EarthformerQML(cfg)

    model = model.to(device)
    model.eval()

    all_pred = []
    all_tgt = []
    
    with torch.no_grad():
        for x, targets in dataloader:
            x = x.to(device)
            y = targets["fire_probability"].to(device)
            preds = model(x)
            pred_probs = torch.sigmoid(preds["fire_probability"])
            
            all_pred.append(pred_probs.cpu())
            all_tgt.append(y.cpu())

    all_pred = torch.cat(all_pred, dim=0).numpy()
    all_tgt = torch.cat(all_tgt, dim=0).numpy()

    mse = np.mean((all_pred - all_tgt) ** 2)
    mae = np.mean(np.abs(all_pred - all_tgt))
    rmse = np.sqrt(mse)

    pred_bin = (all_pred > threshold).astype(np.uint8)
    tgt_bin = (all_tgt > threshold).astype(np.uint8)

    tp = int((pred_bin & tgt_bin).sum())
    fp = int((pred_bin & (1 - tgt_bin)).sum())
    fn = int(((1 - pred_bin) & tgt_bin).sum())

    n_pos = int(tgt_bin.sum())
    if n_pos == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mse": mse, "mae": mae, "rmse": rmse, "tp": tp, "fp": fp, "fn": fn, "n_pos": n_pos}

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1_score, "mse": mse, "mae": mae, "rmse": rmse, "tp": tp, "fp": fp, "fn": fn, "n_pos": n_pos}


# ─────────────────────────────────────────────
# 12. VISUALIZATION FUNCTION (Module Level)
# ─────────────────────────────────────────────

def visualize_prediction(x, y_true, probabilities, binary_prediction, batch_idx=0, time_idx=0):
    """Plots the Ground Truth next to the Model's Probability and Binary Prediction."""
    truth_map = y_true[batch_idx, time_idx, 0].cpu().numpy()
    prob_map  = probabilities[batch_idx, time_idx, 0].cpu().numpy()
    pred_map  = binary_prediction[batch_idx, time_idx, 0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(truth_map, cmap='Reds', vmin=0, vmax=1)
    axes[0].set_title("Ground Truth (Actual Fire)")
    
    heatmap = axes[1].imshow(prob_map, cmap='magma', vmin=0, vmax=1)
    axes[1].set_title("Model Probability Map")
    fig.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
    
    axes[2].imshow(pred_map, cmap='Reds', vmin=0, vmax=1)
    axes[2].set_title("Binary Prediction (Threshold > 0.5)")
    
    plt.tight_layout()
    
    # NEW: Saves the plot to your folder so you always have a copy!
    plt.savefig("prediction_result.png", dpi=300)
    print("Plot saved to prediction_result.png")
    
    plt.show()


# ─────────────────────────────────────────────
# ENTRY POINT (Main Execution Block)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    cfg = Config()
    
    # Reduce spatial dims for quick smoke-test
    cfg.H, cfg.W = 32, 32
    cfg.T, cfg.T_out = 4, 2
    cfg.n_enc_layers = 2
    cfg.n_dec_layers = 2
    cfg.d_model = 64
    cfg.n_heads = 4
    cfg.n_qubits = 4
    cfg.q_depth  = 2
    cfg.q_in_dim = 8

    print("=" * 55)
    print(" Earthformer + QML  |  Wildfire Prediction")
    print("=" * 55)

    model = EarthformerQML(cfg)
    print(f"\nModel built. Encoder layers: {cfg.n_enc_layers}, Decoder layers: {cfg.n_dec_layers}")
    
    print("\nStarting training ...\n")
    model = train_on_trainandtar(cfg=cfg, train_dir="trainandtar", n_epochs=10, batch_size=4)

    print("\nEvaluating model ...\n")
    eval_results = evaluate_trainandtar(cfg=cfg, model=model, train_dir="trainandtar", batch_size=4)
    
    print("\nEvaluation Results:")
    print(f"  Precision: {eval_results['precision']:.4f}")
    print(f"  Recall: {eval_results['recall']:.4f}")
    print(f"  F1 Score: {eval_results['f1']:.4f}")
    print(f"  MSE: {eval_results['mse']:.6f}")
    print(f"  False Negatives: {eval_results['fn']}")

    # ─────────────────────────────────────────────
    # TRIGGER VISUALIZATION
    # ─────────────────────────────────────────────
    print("\nGenerating visual plot...")
    
    X_viz = np.load(os.path.join("trainandtar", "train_input.npy"))
    Y_viz = np.load(os.path.join("trainandtar", "train_target.npy"))
    viz_dataset = TrainAndTarDataset(X_viz, Y_viz, cfg)
    
    viz_dataloader = torch.utils.data.DataLoader(viz_dataset, batch_size=4, shuffle=True)
    
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        x_batch, target_batch = next(iter(viz_dataloader))
        
        x_batch = x_batch.to(device)
        y_true = target_batch["fire_probability"].to(device)
        
        preds = model(x_batch)
        probabilities = torch.sigmoid(preds["fire_probability"])
        binary_prediction = (probabilities > 0.5).int()
        
        # This will now safely trigger the function defined at the module level!
        visualize_prediction(x_batch, y_true, probabilities, binary_prediction, batch_idx=0, time_idx=0)