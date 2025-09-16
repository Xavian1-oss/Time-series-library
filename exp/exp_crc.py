# ./exp/exp_crc.py

import os
import torch
import numpy as np
import warnings

# 导入基础实验类和原有短期预测实验类
from exp.exp_basic import Exp_Basic
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast

# 导入你的 CRC 模型代码
from models.CRC import PretrainableEncoderMTL, HybridTrainer, mtl_pretrain_trainer, build_corr_graph, build_physics_priors

# 导入数据加载和工具函数
from data_provider.data_factory import data_provider
from utils.metrics import metric

warnings.filterwarnings('ignore')

class Exp_CRC(Exp_Basic):
    """
    Causal Residual Corrector (CRC) 实验流程类。
    该类负责管理两阶段的训练和测试流程：
    1. 训练一个基线模型。
    2. 在基线模型的基础上训练一个校正器。
    """
    def __init__(self, args):
        """
        重写 __init__ 方法以绕过父类 Exp_Basic 中不兼容的模型构建逻辑。
        我们在这里手动完成必要的初始化。
        """
        # 手动执行父类中必要的操作，但跳过模型构建
        self.args = args
        # 设备信息在 run.py 中已经设置好并传入 args，我们直接使用
        self.device = args.device

        # CRC 流程专属的初始化
        self.baseline_model = None
        self.hybrid_corrector = None

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _build_model(self):
        # 此方法在此流程中是占位符，因为模型是分阶段动态构建的。
        return None

    def _get_predictions(self, model, data_loader):
        """
        使用指定模型在数据加载器上生成预测。
        新版本：同时返回预测、真实标签和对应的输入窗口，以确保数据完全同步。
        """
        model.eval()
        all_preds = []
        all_trues = []
        all_inputs = []  # <-- 新增：用于存储输入窗口 (batch_x)

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = model(batch_x, None, dec_inp, None)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                all_preds.append(outputs.detach().cpu().numpy())
                all_trues.append(batch_y.detach().cpu().numpy())
                all_inputs.append(batch_x.detach().cpu().numpy()) # <-- 新增：收集输入窗口

        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)
        inputs = np.concatenate(all_inputs, axis=0) # <-- 新增：拼接所有输入窗口
        
        return preds, trues, inputs # <-- 新增：返回三个数组


    def train(self, setting):
        print(">>>>>>> 开始 CRC 两阶段训练流程 <<<<<<<")

        # =================================================================================
        # 阶段一: 训练基线模型
        # =================================================================================
        print("\n" + "=" * 50)
        print("--- 阶段一：训练基线模型 ---")
        print("=" * 50)

        baseline_args = self.args
        baseline_args.model = self.args.baseline_model
        print(f"正在训练基线模型: {baseline_args.model}...")
        baseline_exp = Exp_Short_Term_Forecast(baseline_args)
        self.baseline_model = baseline_exp.train(setting + '_baseline')
        print(f"基线模型训练完成。")

        # =================================================================================
        # 阶段二: 准备数据并训练 CRC 校正器
        # =================================================================================
        print("\n" + "=" * 50)
        print("--- 阶段二：训练 CRC 校正器 ---")
        print("=" * 50)

        train_set, train_loader = self._get_data(flag='train')
        val_set, val_loader = self._get_data(flag='val')

        print("使用基线模型生成预测、真实标签和输入窗口...")
        # <<< 变更点 1：捕获所有三个返回的数组，确保它们完全同步 >>>
        yhat_tr_bl, y_tr_windowed, W_tr_synced = self._get_predictions(self.baseline_model, train_loader)
        yhat_vl_bl, y_vl_windowed, W_vl_synced = self._get_predictions(self.baseline_model, val_loader)

        # 添加调试打印，以确认所有数组的样本数都一致
        print(f"DEBUG: Synced Train Shapes - W_tr: {W_tr_synced.shape}, y_tr: {y_tr_windowed.shape}, yhat_tr: {yhat_tr_bl.shape}")
        print(f"DEBUG: Synced Vali Shapes - W_vl: {W_vl_synced.shape}, y_vl: {y_vl_windowed.shape}, yhat_vl: {yhat_vl_bl.shape}")


        # 1. 构建图邻接矩阵 A
        # <<< 变更点 2：使用同步后的 W_tr_synced 来构建图 >>>
        raw_train_data = train_set.scaler.inverse_transform(W_tr_synced.reshape(-1, W_tr_synced.shape[-1]))
        A = build_corr_graph(raw_train_data, topk=3, thr=0.2).astype(np.float32)
        print(f"邻接矩阵构建完成，密度: {A.sum() / (A.shape[0] ** 2):.3f}")

        # 2. 训练 MTL 编码器
        print("\n--- Stage A: MTL Pre-training ---")
        pre_encoder = PretrainableEncoderMTL(A, self.args.seq_len, self.args.pred_len, self.args.enc_in,
                                            self.args.d_model, self.args.q_val)
        # <<< 变更点 3：使用同步后的 W_tr_synced 和 W_vl_synced >>>
        mtl_pretrain_trainer(pre_encoder, W_tr_synced, y_tr_windowed, W_vl_synced, y_vl_windowed, yhat_tr_bl, yhat_vl_bl)

        # 3. 训练 Hybrid Corrector
        print("\n--- Stage B: Hybrid Corrector ---")
        # <<< 变更点 4：同样使用同步后的 W_tr_synced >>>
        _, F_mu, F_sd = build_physics_priors(W_tr_synced, A, K=24)
        self.hybrid_corrector = HybridTrainer(pre_encoder.encoder, A, F_mu=F_mu, F_sd=F_sd,
                                            H=self.args.pred_len, N=self.args.enc_in)
        # <<< 变更点 5：同样使用同步后的所有数据 >>>
        self.hybrid_corrector.train(W_tr_synced, y_tr_windowed, yhat_tr_bl, W_vl_synced, y_vl_windowed, yhat_vl_bl)

        print("\n>>>>>>> CRC 训练流程全部完成 <<<<<<<")
        return self.hybrid_corrector


    def test(self, setting):
        print(">>>>>>> 开始 CRC 测试流程 <<<<<<<")

        if self.hybrid_corrector is None or self.baseline_model is None:
            raise ValueError("模型尚未训练，请先调用 train() 方法。")

        test_set, test_loader = self._get_data(flag='test')
        
        print("在测试集上生成基线预测、真实标签和输入窗口...")
        # <<< 变更点 1：正确解包 _get_predictions 返回的三个值 >>>
        yhat_te_bl, y_te_synced, W_te_synced = self._get_predictions(self.baseline_model, test_loader)

        print("使用 Hybrid Corrector 对基线预测进行校正...")
        # <<< 变更点 2：使用从 DataLoader 同步获取的 W_te_synced 进行预测 >>>
        yfinal_te = self.hybrid_corrector.predict(W_te_synced, yhat_te_bl)

        # --- 评估和结果保存 ---
        print("\n" + "=" * 50)
        print("--- 最终性能评估 ---")
        print("=" * 50)

        # <<< 变更点 3：使用从 DataLoader 同步获取的 y_te_synced 进行评估 >>>
        # 计算基线模型的结果
        base_mae, base_mse, _, _, _ = metric(yhat_te_bl, y_te_synced)
        print(f"基线模型 ({self.args.baseline_model}) -> MSE: {base_mse:.4f}, MAE: {base_mae:.4f}")

        # 计算 CRC 校正后的结果
        final_mae, final_mse, _, _, _ = metric(yfinal_te, y_te_synced)
        print(f"CRC 校正后模型 -> MSE: {final_mse:.4f}, MAE: {final_mae:.4f}")

        print("-" * 50)
        mae_improvement = (base_mae - final_mae) / base_mae * 100
        mse_improvement = (base_mse - final_mse) / base_mse * 100
        print(f"MAE 相对提升率: {mae_improvement:.2f}%")
        print(f"MSE 相对提升率: {mse_improvement:.2f}%")
        print("=" * 50)

        # 保存结果
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'metrics.npy', np.array([final_mae, final_mse]))
        np.save(folder_path + 'pred.npy', yfinal_te)
        np.save(folder_path + 'true.npy', y_te_synced)
        np.save(folder_path + 'pred_baseline.npy', yhat_te_bl)
        
        return