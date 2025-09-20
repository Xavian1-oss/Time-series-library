from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

# =========================
# Helpers
# =========================
def mae(y, yhat): return float(np.mean(np.abs(y - yhat)))
def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat) ** 2)))
def make_identity_adjacency(N: int) -> np.ndarray: return np.eye(N, dtype=np.int32)

def build_corr_graph(X_tr_raw: np.ndarray, topk: int = 3, thr: Optional[float] = 0.2) -> np.ndarray:
    """Correlation graph on training data; supports (T,N) or (B,P,N)."""
    X = X_tr_raw.reshape(-1, X_tr_raw.shape[-1]) if X_tr_raw.ndim == 3 else X_tr_raw
    C = np.corrcoef(X.T); N = C.shape[0]; A = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        idx = np.argsort(-np.abs(C[i])); cnt = 0
        for j in idx:
            if i == j: continue
            if (thr is None) or (abs(float(C[i,j])) >= thr):
                A[i, j] = 1; cnt += 1
            if cnt >= topk: break
    np.fill_diagonal(A, 1); return A

@dataclass
class GraphCfg:
    N: int = 1; P: int = 96; H: int = 24
    d_model: int = 64; Q: int = 2; seed: int = 42

def sliding_windows(X: np.ndarray, P: int, H: int) -> Tuple[np.ndarray, np.ndarray]:
    T, N = X.shape; B = T - P - H + 1
    if B <= 0: return np.empty((0, P, N), np.float32), np.empty((0, H, N), np.float32)
    W = np.array([X[b:b+P] for b in range(B)], dtype=np.float32)
    y = np.array([X[b+P:b+P+H] for b in range(B)], dtype=np.float32)
    return W, y

# =========================
# Multi-Scale Priors
# =========================
def build_physics_priors_ms(
    W: np.ndarray,
    A: np.ndarray,
    scales: List[int] = [6, 12, 24, 48],
    freq_bands: List[Tuple[float, float]] = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.35)],
    use_robust: bool = True,
    top_m: Optional[int] = None,            # e.g. 32; None = no feature selection
    R_train: Optional[np.ndarray] = None,   # (B,H,N), only used when top_m is set
    mu: Optional[np.ndarray] = None,
    sd: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Return F_norm:(B,N,F), mu:(1,1,F), sd:(1,1,F), feat_names."""
    B, P, N = W.shape; At = A.T.astype(np.float32)
    feats, names = [], []

    def _ema_last_window(x, s):
        alpha = 2.0 / (s + 1.0); xk = x[:, -min(s, P):, :]
        ema = np.zeros((B, N), dtype=np.float32)
        for t in range(xk.shape[1]):
            ema = xk[:, t, :] if t == 0 else alpha * xk[:, t, :] + (1 - alpha) * ema
        return ema.astype(np.float32)

    def _acf_lag(xs, lag):
        if xs.shape[1] <= lag: return np.zeros((xs.shape[0], xs.shape[2]), dtype=np.float32)
        x = xs - xs.mean(axis=1, keepdims=True)
        num = (x[:, lag:, :] * x[:, :-lag, :]).sum(axis=1)
        den = (x * x).sum(axis=1) + 1e-8
        return (num / den).astype(np.float32)

    def _fft_bands(xs, bands):
        Bz, sz, Nz = xs.shape; win = np.hanning(sz).astype(np.float32)[None, :, None]
        xw = xs * win; spec = np.fft.rfft(xw, axis=1)
        power = (spec.real**2 + spec.imag**2) / max(sz, 1)
        freqs = np.fft.rfftfreq(sz, d=1.0)
        out = []
        for lo, hi in bands:
            mask = (freqs >= lo) & (freqs < hi)
            out.append(power[:, mask, :].sum(axis=1).astype(np.float32) if np.any(mask)
                        else np.zeros((Bz, Nz), dtype=np.float32))
        return np.stack(out, axis=-1)  # (B,N,#bands)

    last = W[:, -1, :]; prev = W[:, -2, :] if P >= 2 else W[:, -1, :]
    feats += [last, last - prev, last * prev, last**2]
    names += ["last", "diff1", "prod12", "last_sq"]

    for s in scales:
        s_eff = min(s, P); recent = W[:, -s_eff:, :]
        feats.append(recent.mean(axis=1)); names.append(f"mean_s{s}")
        feats.append(_ema_last_window(W, s_eff)); names.append(f"ema_s{s}")
        feats.append(recent.var(axis=1)); names.append(f"var_s{s}")
        q75 = np.quantile(recent, 0.75, axis=1).astype(np.float32)
        q25 = np.quantile(recent, 0.25, axis=1).astype(np.float32)
        feats.append((q75 - q25)); names.append(f"iqr_s{s}")
        feats.append(np.abs(np.diff(recent, axis=1)).sum(axis=1)); names.append(f"absdiff_energy_s{s}")
        feats.append(_acf_lag(recent, 1)); names.append(f"acf1_s{s}")
        feats.append(_acf_lag(recent, max(2, s_eff // 2))); names.append(f"acf_half_s{s}")
        pb = _fft_bands(recent, freq_bands)  # (B,N,Band)
        for bi in range(pb.shape[-1]):
            feats.append(pb[..., bi]); names.append(f"pband{bi}_s{s}")

    F0 = np.stack(feats, axis=-1).astype(np.float32)  # (B,N,F0)

    # Graph aggregation (A^T)
    Bz, Nz, Fz = F0.shape; Fg = np.empty_like(F0)
    for k in range(Fz): Fg[:, :, k] = F0[:, :, k] @ At  # (B,N)

    # Optional feature selection by residual correlation (per node)
    if (top_m is not None) and (R_train is not None):
        Rm = np.mean(np.abs(R_train), axis=1)  # (B,N)
        F_sel = np.zeros_like(Fg)
        for i in range(N):
            Fi = Fg[:, i, :]; ri = Rm[:, i]
            Fi_c = Fi - Fi.mean(axis=0, keepdims=True); ri_c = ri - ri.mean()
            num = (Fi_c * ri_c[:, None]).sum(axis=0)
            den = (Fi_c**2).sum(axis=0)**0.5 * ((ri_c**2).sum()**0.5 + 1e-8)
            corr = num / (den + 1e-8)
            top_idx = np.argsort(-np.abs(corr))[:top_m]
            mask = np.zeros(Fz, dtype=bool); mask[top_idx] = True
            F_sel[:, i, :] = Fg[:, i, :] * mask[None, :]
        Fg = F_sel

    # Normalization
    if (mu is not None) and (sd is not None):
        F_norm = ((Fg - mu) / (sd + 1e-6)).astype(np.float32)
        return F_norm, mu.astype(np.float32), sd.astype(np.float32), names

    if use_robust:
        med = np.median(Fg, axis=(0,1), keepdims=True)
        mad = np.median(np.abs(Fg - med), axis=(0,1), keepdims=True) + 1e-6
        F_norm = ((Fg - med) / mad).astype(np.float32)
        return F_norm, med.astype(np.float32), mad.astype(np.float32), names
    else:
        mu_out = Fg.mean(axis=(0,1), keepdims=True)
        sd_out = Fg.std(axis=(0,1), keepdims=True) + 1e-6
        F_norm = ((Fg - mu_out) / sd_out).astype(np.float32)
        return F_norm, mu_out.astype(np.float32), sd_out.astype(np.float32), names

# =========================
# Encoder / MTL (uncertainty-weighted)
# =========================
class ResidualBlock(nn.Module):
    def __init__(self, i_d, o_d):
        super().__init__(); self.l = nn.Linear(i_d, o_d)
        self.s = nn.Linear(i_d, o_d) if i_d != o_d else nn.Identity()
    def forward(self, x): return F.relu(self.l(x) + self.s(x))

class STCL_Core(nn.Module):
    def __init__(self, P, d, nl):
        super().__init__()
        self.b = nn.ModuleList([ResidualBlock(P, d)] + [ResidualBlock(d, d) for _ in range(nl - 1)])
        self.g = nn.Parameter(torch.ones(nl))
        self.m = nn.Sequential(nn.Linear(2*d,128), nn.GELU(), nn.Linear(128,4))
    def forward(self, x):
        B = x.shape[0]; A_a = x.new_zeros(B,2,2); r_in = x.permute(0,2,1).contiguous(); R_layers=[]
        for q,blk in enumerate(self.b):
            Rq = blk(r_in); Aq = self.m(Rq.view(B,-1)).view(B,2,2)
            R_layers.append(Aq @ Rq * self.g[q]); A_a += self.g[q]*Aq; r_in = Rq
        return {"R_layers": R_layers, "A_agg": A_a}

class TorchCausalStateEncoder(nn.Module):
    def __init__(self, A, P, d, Q, temp: float = 3.0, self_gate: float = 0.3):
        super().__init__()
        self.register_buffer('A', torch.from_numpy(A.astype(np.float32)))
        self.d = d; self.core = STCL_Core(P, d, Q)
        self.temp, self.self_gate = float(temp), float(self_gate)
        self.edges=[(i,j) for i,row in enumerate(A) for j,val in enumerate(row) if val==1 and i!=j]
        deg=A.sum(axis=1).astype(np.float32)
        self.register_buffer('deg', torch.from_numpy(np.maximum(deg,1e-6)))
    def forward(self, Wb):
        B,P,N=Wb.shape; causal=Wb.new_zeros(B,N,self.d)
        for i in range(N):
            pair=torch.stack([Wb[:,:,i],Wb[:,:,i]],dim=-1); out=self.core(pair)
            R_last, A_agg=out['R_layers'][-1], out['A_agg']
            Ri=F.normalize(R_last[:,0,:],dim=1)
            a_self=torch.tanh(self.temp*A_agg[:,1,0]).unsqueeze(1)
            causal[:,i,:]+=self.self_gate*a_self*Ri
        for (i,j) in self.edges:
            pair=torch.stack([Wb[:,:,i],Wb[:,:,j]],dim=-1); out=self.core(pair)
            R_last, A_agg=out['R_layers'][-1], out['A_agg']
            Rj=F.normalize(R_last[:,1,:],dim=1)
            a21=torch.tanh(self.temp*A_agg[:,1,0]).unsqueeze(1)
            causal[:,i,:]+=a21*Rj
        return causal / (self.deg+self.self_gate).view(1,-1,1)

class PretrainableEncoderMTL(nn.Module):
    def __init__(self, A: np.ndarray, P: int, H: int, N: int, d_model: int = 64, Q: int = 2, recon_hidden: int = 64):
        super().__init__()
        self.encoder = TorchCausalStateEncoder(A, P, d_model, Q)
        self.head_y  = nn.Linear(d_model, H); self.head_dy = nn.Linear(d_model, H)
        self.head_r  = nn.Linear(d_model, H)
        self.recon = nn.Sequential(nn.Linear(d_model, recon_hidden), nn.GELU(), nn.Linear(recon_hidden, 1))
        # learned log variances
        self.log_sigma_y = nn.Parameter(torch.zeros(1))
        self.log_sigma_dy = nn.Parameter(torch.zeros(1))
        self.log_sigma_r = nn.Parameter(torch.zeros(1))
        self.log_sigma_recon = nn.Parameter(torch.zeros(1))
    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0):
        if self.training and mask_ratio > 0:
            K=max(1,int(math.ceil(x.shape[1]*mask_ratio)))
            x_m=x.clone(); x_m[:,-K:,:]=0.0; causal=self.encoder(x_m)
            target_recon=x[:,-K:,:].mean(dim=1); recon_pred=self.recon(causal).squeeze(-1)
        else:
            causal=self.encoder(x); recon_pred, target_recon = None, None
        y_p = self.head_y(causal).permute(0,2,1)
        dy_p= self.head_dy(causal).permute(0,2,1)
        r_p = self.head_r(causal).permute(0,2,1)
        return (y_p, dy_p, r_p, recon_pred, target_recon)

def mtl_pretrain_trainer(model: PretrainableEncoderMTL,
                         W_tr, y_tr, W_vl, y_vl, yhat_tr_bl, yhat_vl_bl,
                         mask_ratio=0.15, lr=1e-3, wd=1e-4, max_epochs=80, patience=10, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"MTL pretrainer using device: {device}")
    model.to(device); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    R_tr = y_tr - yhat_tr_bl
    tr_ds = torch.utils.data.TensorDataset(*(torch.from_numpy(d) for d in [W_tr, y_tr, R_tr]))
    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    Wvl_t, yvl_t = torch.from_numpy(W_vl).to(device), torch.from_numpy(y_vl).to(device)
    best_mae, bad, best_state = float("inf"), 0, None
    for epoch in range(1, max_epochs+1):
        model.train(); total_loss = 0.0
        for wb, yb, Rb in tr_dl:
            wb, yb, Rb = wb.to(device), yb.to(device), Rb.to(device)
            opt.zero_grad()
            y_p, dy_p, r_p, rec_p, rec_t = model(wb, mask_ratio=mask_ratio)
            last_val_wb = wb[:, -1:, :]
            dy_target = yb - last_val_wb
            Ly  = F.mse_loss(y_p, yb)
            Ldy = F.mse_loss(dy_p, dy_target)
            Lr  = F.l1_loss(r_p, Rb)
            Lrec = F.mse_loss(rec_p, rec_t) if rec_p is not None else torch.tensor(0.0, device=device)
            # uncertainty weighting
            loss = (
                torch.exp(-model.log_sigma_y)  * Ly   * 0.5 + model.log_sigma_y * 0.5 +
                torch.exp(-model.log_sigma_dy) * Ldy  * 0.5 + model.log_sigma_dy * 0.5 +
                torch.exp(-model.log_sigma_r)  * Lr   * 0.5 + model.log_sigma_r * 0.5 +
                torch.exp(-model.log_sigma_recon) * Lrec * 0.5 + model.log_sigma_recon * 0.5
            )
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total_loss += float(loss.item())
        model.eval()
        with torch.no_grad():
            yv_p, _, _, _, _ = model(Wvl_t, mask_ratio=0.0)
            mae_v_y = F.l1_loss(yv_p, yvl_t).item()
        print(f"MTL Epoch {epoch:03d}: TrainLoss={total_loss/len(tr_dl):.4f} | Val MAE(y)={mae_v_y:.4f}")
        if mae_v_y < best_mae:
            best_mae, bad, best_state = mae_v_y, 0, deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience: print(f"Early stopping at epoch {epoch}."); break
    if best_state: model.load_state_dict(best_state)


# =========================
# Hybrid Corrector (ridge + shared MLP + tau + blending)
# =========================
def encode_with_encoder(encoder: nn.Module, W: np.ndarray, device: torch.device) -> np.ndarray:
    encoder.eval(); Z_list = []
    with torch.no_grad():
        for b in torch.from_numpy(W).split(512):
            Z_list.append(encoder(b.to(device)).cpu().numpy())
    return np.concatenate(Z_list, axis=0)  # (B,N,d)

def ridge_probe_val(Z_tr, R_tr, Z_vl, yhat_vl, y_vl, l2=1e-2):
    """Per-node ridge for all horizons: Zi_tr:(B,d), Ri_tr:(B,H) -> w:(d,H)"""
    B_tr, H, N = R_tr.shape; _, _, d = Z_tr.shape
    W_list = []; rhat_vl = np.zeros_like(y_vl)
    for i in range(N):
        Zi_tr, Zi_vl = Z_tr[:, i, :], Z_vl[:, i, :]
        Ri_tr = R_tr[:, :, i]  # (B,H)
        A = Zi_tr.T @ Zi_tr + l2 * np.eye(d, dtype=Zi_tr.dtype)      # (d,d)
        b = Zi_tr.T @ Ri_tr                                          # (d,H)
        w = np.linalg.solve(A, b)                                    # (d,H)
        W_list.append(w); rhat_vl[:, :, i] = Zi_vl @ w               # (B,H)
    return mae(y_vl, yhat_vl + rhat_vl), W_list

class SmallMLP(nn.Module):
    def __init__(self, d_in, H, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, H))
    def forward(self, Z): return self.net(Z)  # (B,H)

class HybridTrainer:
    def __init__(self, encoder, A, F_mu, F_sd, H: int, N: int, P: int,
                 l2_delta=1e-4, lr=1e-4, wd=1e-4, patience=20, max_epochs=60, batch_size=64,
                 node_emb_dim: int = 16, use_dir_gate: bool = True,
                 quantile_q: float = 0.80, improve_eps: float = 0.01,
                 scales: List[int] = [6,12,24,48],
                 freq_bands: List[Tuple[float,float]] = [(0.0,0.05),(0.05,0.15),(0.15,0.35)],
                 use_robust: bool = True, top_m: Optional[int] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = deepcopy(encoder).to(self.device).eval()
        for p in self.encoder.parameters(): p.requires_grad = False
        self.A, self.F_mu, self.F_sd, self.H, self.N, self.P = A, F_mu, F_sd, H, N, P
        self.l2_delta, self.lr, self.wd = l2_delta, lr, wd
        self.patience, self.max_epochs, self.batch_size = patience, max_epochs, batch_size
        self.use_dir_gate, self.quantile_q, self.improve_eps = use_dir_gate, quantile_q, improve_eps
        self.scales, self.freq_bands, self.use_robust, self.top_m = scales, freq_bands, use_robust, top_m
        self.W_list_t: List[torch.Tensor] = []; self.tau_map = None   # (H,N)
        self.W_blend = None  # (N,H,2)
        self.linear_is_best = None
        self.node_emb = nn.Embedding(num_embeddings=N, embedding_dim=node_emb_dim).to(self.device)
        self.shared_mlp = SmallMLP(d_in=1, H=H)  # placeholder; reset later

    # ---- feature encoding ----
    def _encode_aug(self, W):
        Z = encode_with_encoder(self.encoder, W, self.device)  # (B,N,d)
        F_aug, _, _, _ = build_physics_priors_ms(
            W, self.A, scales=self.scales, freq_bands=self.freq_bands,
            use_robust=self.use_robust, top_m=self.top_m,
            R_train=None, mu=self.F_mu, sd=self.F_sd
        )  # (B,N,f)
        return np.concatenate([Z, F_aug], axis=-1)  # (B,N,d+f)

    def _init_shared(self, d_aug: int, node_emb_dim: int):
        self.shared_mlp = SmallMLP(d_in=d_aug + node_emb_dim, H=self.H, h=128).to(self.device)

    # ---- residual pieces ----
    def _r_linear(self, Z: torch.Tensor):
        B = Z.shape[0]; r_lin = torch.zeros(B, self.H, self.N, device=self.device)
        for i in range(self.N): r_lin[:, :, i] = Z[:, i, :] @ self.W_list_t[i]  # (B,H)
        return r_lin

    def _r_delta(self, Z: torch.Tensor):
        B = Z.shape[0]; r_delta = torch.zeros(B, self.H, self.N, device=self.device)
        idx = torch.arange(self.N, device=self.device); emb = self.node_emb(idx)  # (N,d_e)
        for i in range(self.N):
            Zi = torch.cat([Z[:, i, :], emb[i].expand(B, -1)], dim=-1)
            r_delta[:, :, i] = self.shared_mlp(Zi)  # (B,H)
        return r_delta

    # ---- safety gate ----
    def _apply_dir_gate(self, r_delta: torch.Tensor, yb: torch.Tensor, yh_b: torch.Tensor,
                        r_lin: torch.Tensor, tau: torch.Tensor):
        if not self.use_dir_gate:
            return torch.clamp(r_delta, -tau, tau)
        err = yb - (yh_b + r_lin)
        same_sign = torch.sign(r_delta) * torch.sign(err) >= 0
        gated = torch.where(same_sign | (torch.abs(r_delta) <= 0.5*tau),
                            r_delta, torch.zeros_like(r_delta))
        return torch.clamp(gated, -tau, tau)

    # ---- training ----
    def train(self, W_tr, y_tr, yhat_tr, W_vl, y_vl, yhat_vl):
        print("[HybridTrainer] Encoding features...")
        # 1) train-time priors statistics
        F_tr_tmp, F_mu, F_sd, _ = build_physics_priors_ms(
            W_tr, self.A, scales=self.scales, freq_bands=self.freq_bands,
            use_robust=self.use_robust, top_m=self.top_m, R_train=(y_tr - yhat_tr),
            mu=None, sd=None
        )
        self.F_mu, self.F_sd = F_mu.astype(np.float32), F_sd.astype(np.float32)

        # 2) encode augmented features
        Z_tr_aug = self._encode_aug(W_tr); Z_vl_aug = self._encode_aug(W_vl)
        d_aug = Z_tr_aug.shape[2]; self._init_shared(d_aug=d_aug, node_emb_dim=self.node_emb.embedding_dim)

        # 3) linear ridge floor
        R_tr = y_tr - yhat_tr
        val_mae_lin, W_list = ridge_probe_val(Z_tr_aug, R_tr, Z_vl_aug, yhat_vl, y_vl, l2=1e-2)
        self.W_list_t = [torch.from_numpy(w.astype(np.float32)).to(self.device) for w in W_list]
        print(f"[Hybrid] Linear-only Val MAE = {val_mae_lin:.4f}")

        # 4) tau from validation residual quantile (per (h,i))
        Zvl_t = torch.from_numpy(Z_vl_aug).to(self.device)
        yvl_t = torch.from_numpy(y_vl).to(self.device)
        yhvl_t = torch.from_numpy(yhat_vl).to(self.device)
        with torch.no_grad():
            r_lin_v = self._r_linear(Zvl_t)
            abs_res = torch.abs(yvl_t - (yhvl_t + r_lin_v))  # (B,H,N)
        tau_np = np.quantile(abs_res.cpu().numpy(), self.quantile_q, axis=0)  # (H,N)
        self.tau_map = torch.from_numpy(tau_np.astype(np.float32)).to(self.device)

        # 5) train shared MLP (with gate + clamp + L2 on delta)
        tr_ds = torch.utils.data.TensorDataset(*(torch.from_numpy(d) for d in [Z_tr_aug, y_tr, yhat_tr]))
        tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.shared_mlp.parameters(), lr=self.lr, weight_decay=self.wd)
        best_mae, best_state, bad_epochs = val_mae_lin, deepcopy(self.shared_mlp.state_dict()), 0
        self.linear_is_best = torch.ones(self.N, self.H, dtype=torch.bool, device=self.device)
        for ep in range(1, self.max_epochs + 1):
            self.shared_mlp.train()
            for Zb, yb, yh_b in tr_dl:
                Zb, yb, yh_b = Zb.to(self.device), yb.to(self.device), yh_b.to(self.device)
                opt.zero_grad()
                r_lin = self._r_linear(Zb.detach())
                r_delta = self._r_delta(Zb)
                tau = self.tau_map.unsqueeze(0).expand_as(r_delta)   # (1,H,N)->(B,H,N)
                r_delta_c = self._apply_dir_gate(r_delta, yb, yh_b, r_lin, tau)
                y_f = yh_b.detach() + r_lin + r_delta_c
                loss = F.l1_loss(y_f, yb) + self.l2_delta * (r_delta_c**2).mean()
                loss.backward(); opt.step()

            # validate (per (h,i) choose better)
            self.shared_mlp.eval()
            with torch.no_grad():
                r_lin_v = self._r_linear(Zvl_t)
                r_delta_v = self._r_delta(Zvl_t)
                tau_v = self.tau_map.unsqueeze(0).expand_as(r_delta_v)
                r_delta_v = torch.clamp(r_delta_v, -tau_v, tau_v)
                y_lin = yhvl_t + r_lin_v
                y_hyb = y_lin + r_delta_v
                err_lin = torch.mean(torch.abs(yvl_t - y_lin), dim=0)   # (H,N)
                err_hyb = torch.mean(torch.abs(yvl_t - y_hyb), dim=0)   # (H,N)
                better_hyb = (err_hyb + 1e-8) < (err_lin * (1 - self.improve_eps))
                self.linear_is_best = (~better_hyb).permute(1,0).contiguous()  # (N,H)
                mae_v_mix = torch.mean(torch.where(better_hyb, err_hyb, err_lin)).item()
            print(f"[Hybrid] Epoch {ep:03d}: Val MAE (mixed)={mae_v_mix:.4f}")
            if mae_v_mix < best_mae:
                best_mae, best_state, bad_epochs = mae_v_mix, deepcopy(self.shared_mlp.state_dict()), 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    print(f"Early stopping at epoch {ep}."); break
        if best_state: self.shared_mlp.load_state_dict(best_state)

        # 6) per-(node,h) blending weights grid search (shrink-to-base)
        print("[Hybrid] Calibrating per-(node,h) blending weights ...")
        with torch.no_grad():
            r_lin_v = self._r_linear(Zvl_t)
            r_delta_v = self._r_delta(Zvl_t)
            tau_v = self.tau_map.unsqueeze(0).expand_as(r_delta_v)
            r_delta_v = torch.clamp(r_delta_v, -tau_v, tau_v)
            base = yhvl_t; y_true = yvl_t
            W_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
            self.W_blend = torch.zeros(self.N, self.H, 2, device=self.device)
            base_mae_map = torch.mean(torch.abs(y_true - base), dim=0)  # (H,N)
            for i in range(self.N):
                for h in range(self.H):
                    best_mae_ih = base_mae_map[h,i].item(); best_w = (0.0, 0.0)
                    for w1 in W_grid:
                        for w2 in W_grid:
                            y_hat = base[:,h,i] + w1 * r_lin_v[:,h,i] + w2 * r_delta_v[:,h,i]
                            m = torch.mean(torch.abs(y_true[:,h,i] - y_hat)).item()
                            if m < best_mae_ih - 1e-12: best_mae_ih, best_w = m, (w1, w2)
                    # shrink-to-base if improvement not significant
                    if best_mae_ih > base_mae_map[h,i].item() * (1 - self.improve_eps):
                        best_w = (0.0, 0.0)
                    self.W_blend[i,h,0], self.W_blend[i,h,1] = best_w[0], best_w[1]
        print("[Hybrid] Blending calibration done.")

    # ---- inference ----
    def predict(self, W: np.ndarray, yhat: np.ndarray):
        Z_aug = self._encode_aug(W)
        Zt = torch.from_numpy(Z_aug).to(self.device)
        yh_t = torch.from_numpy(yhat).to(self.device)
        with torch.no_grad():
            r_lin = self._r_linear(Zt)
            r_delta = self._r_delta(Zt)
            tau = self.tau_map.unsqueeze(0).expand_as(r_delta)
            r_delta_c = torch.clamp(r_delta, -tau, tau)
            w1 = self.W_blend[:,:,0].permute(1,0).unsqueeze(0)  # (1,H,N)
            w2 = self.W_blend[:,:,1].permute(1,0).unsqueeze(0)
            y_final = yh_t + w1 * r_lin + w2 * r_delta_c
        return y_final.cpu().numpy()
