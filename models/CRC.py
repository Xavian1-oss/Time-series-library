# ./models/CRC.py

"""
Causal Residual Corrector (CRC) v12.3 - 模型核心组件模块

该文件包含了 CRC 工作流所需的所有核心模型、训练器和辅助函数。
它被设计为可被外部实验框架导入的模块。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


# ===============================================
# Helper Functions (通用辅助函数)
# ===============================================

def mae(y, yhat):
    """计算 Mean Absolute Error (MAE)"""
    return float(np.mean(np.abs(y - yhat)))

def rmse(y, yhat):
    """计算 Root Mean Squared Error (RMSE)"""
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def make_identity_adjacency(N: int) -> np.ndarray:
    """创建一个单位邻接矩阵"""
    return np.eye(N, dtype=np.int32)

def build_corr_graph(X_tr_raw: np.ndarray, topk: int = 3, thr: float = 0.2) -> np.ndarray:
    """根据特征相关性构建邻接矩阵"""
    C = np.corrcoef(X_tr_raw.T)
    N = C.shape[0]
    A = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        idx = np.argsort(-np.abs(C[i]))
        cnt = 0
        for j in idx:
            if i == j: continue
            if abs(float(C[i, j])) >= thr:
                A[i, j] = 1
                cnt += 1
            if cnt >= topk: break
    np.fill_diagonal(A, 1)
    return A

def build_physics_priors(W: np.ndarray, A: np.ndarray, K: int = 24,
                         mu: Optional[np.ndarray] = None,
                         sd: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建基于物理先验的特征"""
    B, P, N = W.shape; At = A.T.astype(np.float32)
    last, prev = W[:, -1, :], W[:, -2, :]
    neigh_t1 = last @ At; diff_1 = (last - prev) @ At; prod_12 = (last * prev) @ At
    recent = W[:, -K:, :]; mov_avg = recent.mean(axis=1) @ At
    hp_energy = np.abs(np.diff(recent, axis=1)).sum(axis=1) @ At
    q75 = np.quantile(recent, 0.75, axis=1).astype(np.float32)
    q25 = np.quantile(recent, 0.25, axis=1).astype(np.float32)
    iqr_k = (q75 - q25); var_k = recent.var(axis=1); last_sq_neigh = (last**2) @ At
    F = np.stack([neigh_t1, diff_1, prod_12, mov_avg, hp_energy, iqr_k, var_k, last_sq_neigh], axis=-1)
    if mu is None or sd is None:
        mu = F.mean(axis=(0,1), keepdims=True)
        sd = F.std(axis=(0,1), keepdims=True) + 1e-6
    return ((F - mu) / sd).astype(np.float32), mu, sd

# ===============================================
# Core CRC Model Components (CRC 核心模型组件)
# ===============================================

class ResidualBlock(nn.Module):
    def __init__(self, i_d, o_d):
        super().__init__(); self.l = nn.Linear(i_d, o_d); self.s = nn.Linear(i_d, o_d) if i_d != o_d else nn.Identity()
    def forward(self, x): return F.relu(self.l(x) + self.s(x))

class STCL_Core(nn.Module):
    def __init__(self, P, d, nl):
        super().__init__(); self.b = nn.ModuleList([ResidualBlock(P, d)] + [ResidualBlock(d, d) for _ in range(nl - 1)]); self.g = nn.Parameter(torch.ones(nl)); self.m = nn.Sequential(nn.Linear(2*d,128), nn.GELU(), nn.Linear(128,4))
    def forward(self, x):
        B = x.shape[0]; A_a = x.new_zeros(B,2,2); r_in = x.permute(0,2,1).contiguous(); R_layers = []
        for q, blk in enumerate(self.b):
            Rq = blk(r_in); Aq = self.m(Rq.view(B,-1)).view(B,2,2); R_layers.append(Aq @ Rq * self.g[q]); A_a += self.g[q] * Aq; r_in = Rq
        return {"R_layers": R_layers, "A_agg": A_a}

class TorchCausalStateEncoder(nn.Module):
    def __init__(self, A, P, d, Q, temp: float = 3.0, self_gate: float = 0.3):
        super().__init__(); self.register_buffer('A', torch.from_numpy(A.astype(np.float32))); self.d = d; self.core = STCL_Core(P, d, Q); self.temp, self.self_gate = float(temp), float(self_gate); self.edges=[(i,j) for i, row in enumerate(A) for j, val in enumerate(row) if val==1 and i!=j]; deg=A.sum(axis=1).astype(np.float32); self.register_buffer('deg', torch.from_numpy(np.maximum(deg,1e-6)))
    def forward(self, Wb):
        B,P,N=Wb.shape; causal=Wb.new_zeros(B,N,self.d)
        for i in range(N):
            pair=torch.stack([Wb[:,:,i],Wb[:,:,i]],dim=-1); out=self.core(pair); R_last, A_agg=out['R_layers'][-1], out['A_agg']; Ri=F.normalize(R_last[:,0,:],dim=1); a_self=torch.tanh(self.temp*A_agg[:,1,0]).unsqueeze(1); causal[:,i,:]+=self.self_gate*a_self*Ri
        for (i,j) in self.edges:
            pair=torch.stack([Wb[:,:,i],Wb[:,:,j]],dim=-1); out=self.core(pair); R_last, A_agg=out['R_layers'][-1], out['A_agg']; Rj=F.normalize(R_last[:,1,:],dim=1); a21=torch.tanh(self.temp*A_agg[:,1,0]).unsqueeze(1); causal[:,i,:]+=a21*Rj
        return causal / (self.deg+self.self_gate).view(1,-1,1)

class PretrainableEncoderMTL(nn.Module):
    def __init__(self, A: np.ndarray, P: int, H: int, N: int, d_model: int, Q: int, recon_hidden: int = 64):
        super().__init__()
        self.encoder = TorchCausalStateEncoder(A, P, d_model, Q)
        self.head_y  = nn.Linear(d_model, H); self.head_dy = nn.Linear(d_model, H)
        self.head_r  = nn.Linear(d_model, H)
        self.recon = nn.Sequential(nn.Linear(d_model, recon_hidden), nn.GELU(), nn.Linear(recon_hidden, 1))

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0):
        if self.training and mask_ratio > 0:
            K = max(1, int(math.ceil(x.shape[1] * mask_ratio)))
            x_m = x.clone(); x_m[:, -K:, :] = 0.0; causal = self.encoder(x_m)
            target_recon = x[:, -K:, :].mean(dim=1); recon_pred = self.recon(causal).squeeze(-1)
        else:
            causal = self.encoder(x); recon_pred, target_recon = None, None
        y_p = self.head_y(causal).permute(0, 2, 1)
        dy_p = self.head_dy(causal).permute(0, 2, 1)
        r_p = self.head_r(causal).permute(0, 2, 1)
        return (y_p, dy_p, r_p, recon_pred, target_recon)

def mtl_pretrain_trainer(model: PretrainableEncoderMTL,
                         W_tr, y_tr, W_vl, y_vl, yhat_tr_bl, yhat_vl_bl,
                         alpha_dy=0.3, alpha_recon=0.15, alpha_r=1.5, mask_ratio=0.15,
                         lr=1e-3, wd=1e-5, max_epochs=80, patience=10, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"MTL pretrainer using device: {device}")
    model.to(device); opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
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
            loss = (F.mse_loss(y_p, yb) + alpha_dy * F.mse_loss(dy_p, dy_target) + alpha_r * F.l1_loss(r_p, Rb))
            if rec_p is not None: loss += alpha_recon * F.mse_loss(rec_p, rec_t)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total_loss += loss.item()
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


# ===============================================
# Hybrid Corrector (混合校正器)
# ===============================================

def encode_with_encoder(encoder: nn.Module, W: np.ndarray, device: torch.device) -> np.ndarray:
    encoder.eval(); Z_list = []
    with torch.no_grad():
        for b in torch.from_numpy(W).split(512):
            Z_list.append(encoder(b.to(device)).cpu().numpy())
    return np.concatenate(Z_list, axis=0)

def ridge_probe_val(Z_tr, R_tr, Z_vl, yhat_vl, y_vl, l2=1e-2) -> Tuple[float, List[np.ndarray]]:
    B_tr, H, N = R_tr.shape
    _, _, d = Z_tr.shape
    W_list = []; rhat_vl = np.zeros_like(y_vl)
    for i in range(N):
        Zi_tr, Zi_vl = Z_tr[:, i, :], Z_vl[:, i, :]
        Ri_tr = R_tr[:, :, i]
        A = Zi_tr.T @ Zi_tr + l2 * np.eye(d, dtype=Zi_tr.dtype)
        b = Zi_tr.T @ Ri_tr
        w = np.linalg.solve(A, b)
        W_list.append(w)
        rhat_vl[:, :, i] = Zi_vl @ w
    return mae(y_vl, yhat_vl + rhat_vl), W_list

class SmallMLP(nn.Module):
    def __init__(self, d_in, H, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h),
            nn.GELU(),
            nn.Linear(h, H)
        )
    def forward(self, Z): 
        return self.net(Z)

class HybridTrainer:
    def __init__(self, encoder, A, F_mu, F_sd, H: int, N: int, P: int,
                 k_trust=2.0, tau_min=0.05, l2_delta=1e-4, lr=1e-4, wd=1e-4, patience=20, max_epochs=60, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = deepcopy(encoder).to(self.device).eval()
        for p in self.encoder.parameters(): p.requires_grad = False
        self.A, self.F_mu, self.F_sd, self.H, self.N, self.P = A, F_mu, F_sd, H, N, P
        self.k_trust, self.tau_min, self.l2_delta, self.lr, self.wd = k_trust, tau_min, l2_delta, lr, wd
        self.patience, self.max_epochs, self.batch_size = patience, max_epochs, batch_size
        self.mlps = None
        self.W_list_t, self.std_node_t, self.use_linear_only = None, None, False

    def _encode_aug(self, W):
        Z = encode_with_encoder(self.encoder, W, self.device)
        F, _, _ = build_physics_priors(W, self.A, K=24, mu=self.F_mu, sd=self.F_sd)
        return np.concatenate([Z, F], axis=-1)

    def _r_linear(self, Z):
        B = Z.shape[0]
        r_lin = torch.zeros(B, self.H, self.N, device=self.device)
        for i in range(self.N):
            r_lin[:, :, i] = Z[:, i, :] @ self.W_list_t[i]
        return r_lin

    def _r_delta(self, Z):
        B = Z.shape[0]
        r_delta = torch.zeros(B, self.H, self.N, device=self.device)
        for i in range(self.N):
            r_delta[:, :, i] = self.mlps[i](Z[:, i, :])
        return r_delta

    def _tau(self):
        return (self.k_trust * self.std_node_t + self.tau_min)

    def _predict_with_best_model(self, Z, yhat):
        r_lin = self._r_linear(Z)
        if self.use_linear_only or self.mlps is None:
            return yhat + r_lin
        
        r_delta = self._r_delta(Z)
        tau = self._tau()
        return yhat + r_lin + torch.clamp(r_delta, -tau, tau)

    def train(self, W_tr, y_tr, yhat_tr, W_vl, y_vl, yhat_vl):
        print("[HybridTrainer] Encoding features...")
        Z_tr_aug, Z_vl_aug = self._encode_aug(W_tr), self._encode_aug(W_vl)
        d_aug = Z_tr_aug.shape[2]; R_tr = y_tr - yhat_tr
        
        val_mae_lin, W_list = ridge_probe_val(Z_tr_aug, R_tr, Z_vl_aug, yhat_vl, y_vl, l2=1e-2)
        self.W_list_t = [torch.from_numpy(w.astype(np.float32)).to(self.device) for w in W_list]
        print(f"[Hybrid] Linear-only Val MAE = {val_mae_lin:.4f}")

        self.mlps = nn.ModuleList([SmallMLP(d_aug, self.H, h=128) for _ in range(self.N)]).to(self.device)
        self.std_node_t = torch.from_numpy(R_tr.std(axis=(0, 1), keepdims=True).astype(np.float32)).to(self.device)
        
        tr_ds = torch.utils.data.TensorDataset(*(torch.from_numpy(d) for d in [Z_tr_aug, y_tr, yhat_tr]))
        tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.mlps.parameters(), lr=self.lr, weight_decay=self.wd)
        Zvl_t, yvl_t, yhvl_t = (torch.from_numpy(d).to(self.device) for d in [Z_vl_aug, y_vl, yhat_vl])

        best_mae, best_mlp_state, self.use_linear_only, bad_epochs = val_mae_lin, deepcopy(self.mlps.state_dict()), True, 0
        print("[Hybrid] Training non-linear delta corrector...")
        for ep in range(1, self.max_epochs + 1):
            self.mlps.train()
            for Zb, yb, yh_b in tr_dl:
                Zb, yb, yh_b = Zb.to(self.device), yb.to(self.device), yh_b.to(self.device)
                opt.zero_grad()
                
                r_lin = self._r_linear(Zb.detach())
                r_delta = self._r_delta(Zb)
                
                tau = self._tau()
                r_delta_clamped = torch.clamp(r_delta, -tau, tau)
                
                y_f = yh_b.detach() + r_lin + r_delta_clamped
                loss = F.l1_loss(y_f, yb) + self.l2_delta * (r_delta_clamped**2).mean()
                loss.backward(); opt.step()

            self.mlps.eval()
            with torch.no_grad():
                y_vf_h = self._predict_with_best_model(Zvl_t, yhvl_t)
                mae_v_h = F.l1_loss(y_vf_h, yvl_t).item()
            print(f"[Hybrid] Epoch {ep:03d}: Val MAE (hybrid)={mae_v_h:.4f} (best was {best_mae:.4f})")
            if mae_v_h < best_mae:
                best_mae, best_mlp_state, self.use_linear_only, bad_epochs = mae_v_h, deepcopy(self.mlps.state_dict()), False, 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience: print(f"Early stopping at epoch {ep}."); break

        if best_mlp_state: self.mlps.load_state_dict(best_mlp_state)
        print(f"[Hybrid] Best choice: {'Linear-only' if self.use_linear_only else 'Hybrid'} with Val MAE={best_mae:.4f}")

    def predict(self, W, yhat):
        if self.mlps is None and not self.use_linear_only:
             raise RuntimeError("Trainer has not been trained or linear-only was chosen but MLP is being used.")
        
        Z_aug = self._encode_aug(W)
        with torch.no_grad():
            y_final = self._predict_with_best_model(
                torch.from_numpy(Z_aug).to(self.device),
                torch.from_numpy(yhat).to(self.device)
            )
        return y_final.cpu().numpy()