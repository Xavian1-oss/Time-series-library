import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Helper: simple Transformer encoder block
# -----------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, dim_feedforward=256, num_layers=2, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
    def forward(self, x):
        # x: (B, T, D)
        return self.encoder(x)

# -----------------------------
# Uncertainty Estimator (Section 3.2): predicts MSE(y_bar, y)
# Input: flattened x (B, N*Lin), flattened y_bar (B, N*Lout), channel id embedding (B, N, d)
# -----------------------------
class UncertaintyEstimator(nn.Module):
    def __init__(self, n_channels, lin, lout, d_embed=64, hidden=256):
        super().__init__()
        self.E = nn.Embedding(n_channels, d_embed)  # channel-identity embeddings
        in_dim = n_channels*lin + n_channels*lout + n_channels*d_embed
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)  # predict scalar error proxy per instance
        )
    def forward(self, x, y_bar):
        # x: (B, N, Lin), y_bar: (B, N, Lout)
        B, N, Lin = x.shape
        _, _, Lout = y_bar.shape
        E = self.E(torch.arange(N, device=x.device)).unsqueeze(0).expand(B, N, -1)  # (B,N,d)
        feat = torch.cat([x.reshape(B, -1), y_bar.reshape(B, -1), E.reshape(B, -1)], dim=1)
        delta = self.mlp(feat)  # (B,1)
        return delta.squeeze(-1)  # (B,)

# -----------------------------
# Local Revising (Section 3.3): use covariate (other channels) forecasts + exogenous as context
# We'll create a token sequence: [per-channel y_bar tokens ... ; exo token]
# Each channel token = linear projection of its horizon vector
# Exo token = projection of exogenous vector for the horizon (e.g., time features)
# -----------------------------
class LocalReviser(nn.Module):
    def __init__(self, n_channels, lout, exo_dim, d_model=128, num_layers=2, nhead=4, ff=256):
        super().__init__()
        self.proj_channel = nn.Linear(lout, d_model)
        self.proj_exo = nn.Linear(exo_dim, d_model)
        self.encoder = TransformerEncoder(d_model=d_model, nhead=nhead, dim_feedforward=ff, num_layers=num_layers)
        self.head = nn.Linear(d_model, lout)  # predict a correction for target channel token
        # We will pool the first token (target channel) after encoding
    def forward(self, y_bar, exo, target_channel=0):
        # y_bar: (B,N,Lout) predicted horizon for each channel
        # exo: (B,exo_dim) exogenous known-in-advance for this horizon/instance
        B, N, L = y_bar.shape
        tokens = [self.proj_channel(y_bar[:,i,:]) for i in range(N)]  # list of (B,D)
        tokens = torch.stack(tokens, dim=1)  # (B,N,D)
        exo_tok = self.proj_exo(exo).unsqueeze(1)  # (B,1,D)
        seq = torch.cat([tokens, exo_tok], dim=1)  # (B,N+1,D)
        enc = self.encoder(seq)  # (B,N+1,D)
        tgt = enc[:, target_channel, :]  # take target channel's contextualized embedding
        y_local = self.head(tgt)  # (B,L)
        return y_local

# -----------------------------
# Global Revising (Section 3.4): retrieval on training (x,y) pairs.
# Enc = instance norm + flatten; similarity = cosine; weighted sum of Y_train.
# -----------------------------
class GlobalRetriever(nn.Module):
    def __init__(self, topk=20):
        super().__init__()
        self.topk = topk
        # buffers to hold retrieval db (filled after fit())
        self.register_buffer("X_ref", None, persistent=False)   # (M, N*Lin)
        self.register_buffer("Y_ref", None, persistent=False)   # (M, N*Lout) or target only (M,Lout)
    @staticmethod
    def _enc(x):
        # x: (B,N,Lin) -> (B, N*Lin), instance-norm per channel then flatten
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-6
        xn = (x - mean) / std
        return xn.reshape(x.shape[0], -1)  # flatten
    def build(self, X_train, Y_train_target):
        # X_train: (M, N, Lin), Y_train_target: (M, Lout) // only target channel for simplicity
        with torch.no_grad():
            self.X_ref = self._enc(X_train).detach()
            self.Y_ref = Y_train_target.detach()
    def retrieve(self, x):
        # x: (B,N,Lin) -> returns (B, topk) weights and retrieved Ys: (B, topk, Lout)
        B = x.shape[0]
        q = self._enc(x)  # (B, D)
        # cosine similarity with ref (M, D)
        # normalize
        qn = F.normalize(q, dim=1)
        rn = F.normalize(self.X_ref, dim=1)  # (M,D)
        sims = torch.matmul(qn, rn.t())  # (B,M)
        topk = min(self.topk, sims.shape[1])
        vals, idx = torch.topk(sims, k=topk, dim=1, largest=True, sorted=True)  # (B,topk)
        Yk = self.Y_ref[idx]  # (B,topk,Lout)
        weights = F.softmax(vals, dim=1)  # (B,topk)
        return weights, Yk

# -----------------------------
# PIR Wrapper (Section 3.5): y_pred = y_bar + alpha*y_local + beta*y_global
# alpha = sigmoid(linear(delta)); beta = sigmoid(MLP(delta, w))
# -----------------------------
class PIRWrapper(nn.Module):
    def __init__(self, backbone, n_channels, lin, lout, exo_dim, topk=20, lambda_aux=1.0, target_channel=0):
        super().__init__()
        self.backbone = backbone
        self.n_channels = n_channels
        self.lin = lin
        self.lout = lout
        self.lambda_aux = lambda_aux
        self.target_channel = target_channel
        self.uncert = UncertaintyEstimator(n_channels, lin, lout)
        self.local = LocalReviser(n_channels, lout, exo_dim)
        self.global_ret = GlobalRetriever(topk=topk)
        # alpha, beta heads
        self.alpha_head = nn.Linear(1, 1)  # input: delta (scalar), init weight=1, bias=0
        with torch.no_grad():
            self.alpha_head.weight.fill_(1.0)
            self.alpha_head.bias.zero_()
        self.beta_head = nn.Sequential(
            nn.Linear(1 + topk, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def build_retrieval(self, X_train, Y_train):
        # Use only target channel for Y_ref (B, Lout)
        Yt = Y_train[:, self.target_channel, :]  # (M,Lout)
        self.global_ret.build(X_train, Yt)
    def forward(self, x, exo, y=None):
        # x: (B,N,Lin), exo: (B, exo_dim)
        # backbone must output per-channel horizon forecasts: (B,N,Lout)
        y_bar = self.backbone(x, exo)  # (B,N,Lout)
        # Uncertainty estimation
        delta = self.uncert(x, y_bar)  # (B,)
        # Local revision -> correction for target channel
        y_local = self.local(y_bar, exo, target_channel=self.target_channel)  # (B,Lout)
        # Global revision via retrieval
        w, Yk = self.global_ret.retrieve(x)  # (B,topk), (B,topk,Lout)
        y_global = torch.sum(w.unsqueeze(-1) * Yk, dim=1)  # (B,Lout)
        # Weights
        alpha = torch.sigmoid(self.alpha_head(delta.unsqueeze(-1))).squeeze(-1)  # (B,)
        beta = torch.sigmoid(self.beta_head(torch.cat([delta.unsqueeze(-1), w], dim=1))).squeeze(-1)  # (B,)
        # Combine
        y_pred_t = y_bar[:, self.target_channel, :] + alpha.unsqueeze(-1)*y_local + beta.unsqueeze(-1)*y_global
        # If training with ground truth y, compute losses
        out = {"y_bar": y_bar, "y_pred_t": y_pred_t, "delta": delta, "alpha": alpha, "beta": beta, "y_local": y_local, "y_global": y_global}
        if y is not None:
            # y: (B,N,Lout); take true target
            y_t = y[:, self.target_channel, :]
            loss_main = F.mse_loss(y_pred_t, y_t)
            with torch.no_grad():
                mse_bar_all = F.mse_loss(y_bar, y, reduction="none").mean(dim=(1,2))  # (B,)
            loss_aux = F.l1_loss(delta, mse_bar_all)
            out["loss"] = loss_main + self.lambda_aux * loss_aux
            out["loss_main"] = loss_main
            out["loss_aux"] = loss_aux
            out["y_t"] = y_t
        return out