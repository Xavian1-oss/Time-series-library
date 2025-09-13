import torch
import torch.nn as nn

class BackboneAdapter(nn.Module):
    """
    Adapter to make backbone models compatible with PIRWrapper.
    """
    def __init__(self, model, Lin, Lout, N):
        super().__init__()
        self.model = model
        self.Lin = Lin
        self.Lout = Lout
        self.N = N

    def forward(self, x, exo=None):
        # x: (B,N,Lin) → TSLib 模型一般需要 (B,Lin,N)
        x_enc = x.transpose(1, 2).contiguous()  # (B, Lin, N)
        # Decoder 输入（全零即可）
        B = x.shape[0]
        x_dec = torch.zeros(B, self.Lout, self.N, device=x.device)
        x_mark_enc = torch.zeros(B, self.Lin, 4, device=x.device)  # 时间特征占位
        x_mark_dec = torch.zeros(B, self.Lout, 4, device=x.device)

        # 调用原模型
        y_hat = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # 常见签名

        # 转为 (B,N,Lout)
        if y_hat.shape[1] == self.Lout:
            return y_hat.transpose(1, 2).contiguous()
        return y_hat