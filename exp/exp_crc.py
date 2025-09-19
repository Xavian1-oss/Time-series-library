import os
import torch
import numpy as np
import warnings
from copy import deepcopy
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from models.CRC import PretrainableEncoderMTL, HybridTrainer, mtl_pretrain_trainer, build_corr_graph, build_physics_priors
from data_provider.data_factory import data_provider
from utils.metrics import metric
from utils.dtw_metric import accelerated_dtw
from utils.tools import visual

warnings.filterwarnings('ignore')


class Exp_CRC(Exp_Long_Term_Forecast):
    """
    Causal Residual Corrector (CRC) 实验流程类。
    该类现在继承自 Exp_Long_Term_Forecast，以更好地匹配其长周期预测任务的性质。
    """
    def __init__(self, args):
        """
        关键修复：我们不再调用 super().__init__(args)。

        父类的初始化链条 (Exp_Basic -> Exp_Long_Term_Forecast) 强制要求立即构建模型，
        而 CRC 在初始化阶段不支持这样做。
        
        我们继承自 Exp_Long_Term_Forecast 是为了它其他有用的方法（比如通用的 test 方法），
        但我们需要一个自定义的初始化流程。我们在这里手动执行最核心的设置。
        """
        self.args = args
        self.device = args.device

        self.baseline_model = None
        self.hybrid_corrector = None
    
    def _build_model(self):
        """
        关键修复：重写此方法以覆盖父类的行为。
        
        这可以防止父类在初始化时，尝试去构建一个名为 "CRC" 的、实际上并不存在的单一模型。
        CRC的真实模型（基线和校正器）是在 train 方法中被动态、分阶段构建的。
        
        我们在这里返回 None 是安全的，因为在CRC的流程中，我们从不使用 self.model 这个属性，
        而是使用我们自己定义的 self.baseline_model 和 self.hybrid_corrector。
        """
        return None

    def _get_predictions(self, model, data_loader):
        """
        使用指定模型在数据加载器上生成预测。
        同时返回预测、真实标签和对应的输入窗口，以确保数据完全同步。
        """
        model.eval()
        all_preds, all_trues, all_inputs = [], [], []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # decoder input (这部分逻辑保持不变，因为它适用于大多数 Encoder-Decoder 模型)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 注意：这里的模型调用没有 x_mark_...，因为CRC的encoder不使用它们。
                # 这与CRC的原始设计一致。
                outputs = model(batch_x, None, dec_inp, None)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                all_preds.append(outputs.detach().cpu().numpy())
                all_trues.append(batch_y.detach().cpu().numpy())
                all_inputs.append(batch_x.detach().cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)
        inputs = np.concatenate(all_inputs, axis=0)
        
        return preds, trues, inputs


    def train(self, setting):
        print(">>>>>>> 开始 CRC 两阶段训练流程 <<<<<<<")

        # =================================================================================
        # 阶段一: 训练基线模型
        # =================================================================================
        print("\n" + "=" * 50)
        print("--- 阶段一：训练基线模型 ---")
        print("=" * 50)
        
        # 7. 改进的健壮性：使用 deepcopy 来创建独立的 baseline_args。
        #    - 这可以防止修改 baseline_args.model 时意外地污染主流程的 self.args。
        baseline_args = deepcopy(self.args)
        baseline_args.model = self.args.baseline_model # 指定要训练的基线模型名称
        
        print(f"正在训练基线模型: {baseline_args.model}...")
        baseline_exp = Exp_Long_Term_Forecast(baseline_args)
        self.baseline_model = baseline_exp.train(setting + '_baseline')
        print(f"基线模型训练完成。")

        # =================================================================================
        # 阶段二: 准备数据并训练 CRC 校正器 (这部分的核心逻辑是CRC独有的，保持不变)
        # =================================================================================
        print("\n" + "=" * 50)
        print("--- 阶段二：训练 CRC 校正器 ---")
        print("=" * 50)

        train_set, train_loader = self._get_data(flag='train')
        val_set, val_loader = self._get_data(flag='val')

        print("使用基线模型生成预测、真实标签和输入窗口...")
        yhat_tr_bl, y_tr_windowed, W_tr_synced = self._get_predictions(self.baseline_model, train_loader)
        yhat_vl_bl, y_vl_windowed, W_vl_synced = self._get_predictions(self.baseline_model, val_loader)

        print(f"DEBUG: Synced Train Shapes - W_tr: {W_tr_synced.shape}, y_tr: {y_tr_windowed.shape}, yhat_tr: {yhat_tr_bl.shape}")
        print(f"DEBUG: Synced Vali Shapes - W_vl: {W_vl_synced.shape}, y_vl: {y_vl_windowed.shape}, yhat_vl: {yhat_vl_bl.shape}")

        # 1. 构建图
        raw_train_data_for_graph = train_set.scaler.inverse_transform(W_tr_synced.reshape(-1, W_tr_synced.shape[-1]))
        A = build_corr_graph(raw_train_data_for_graph, topk=self.args.top_k, thr=0.2).astype(np.float32)
        print(f"邻接矩阵构建完成，密度: {A.sum() / (A.shape[0] ** 2):.3f}")

        # 2. 预训练MTL编码器
        print("\n--- Stage A: MTL Pre-training ---")
        pre_encoder = PretrainableEncoderMTL(A, self.args.seq_len, self.args.pred_len, self.args.enc_in,
                                            self.args.d_model, self.args.q_val)
        mtl_pretrain_trainer(pre_encoder, W_tr_synced, y_tr_windowed, W_vl_synced, y_vl_windowed, yhat_tr_bl, yhat_vl_bl)

        # 3. 训练混合校正器
        print("\n--- Stage B: Hybrid Corrector ---")
        _, F_mu, F_sd = build_physics_priors(W_tr_synced, A, K=self.args.k_val)
        self.hybrid_corrector = HybridTrainer(pre_encoder.encoder, A, F_mu=F_mu, F_sd=F_sd,
                                            H=self.args.pred_len, N=self.args.enc_in, P=self.args.seq_len) # 传入P以支持自适应先验
        self.hybrid_corrector.train(W_tr_synced, y_tr_windowed, yhat_tr_bl, W_vl_synced, y_vl_windowed, yhat_vl_bl)

        print("\n>>>>>>> CRC 训练流程全部完成 <<<<<<<")
        return self.hybrid_corrector


    def test(self, setting):
        print(">>>>>>> 开始 CRC 测试流程 <<<<<<<")

        if self.hybrid_corrector is None or self.baseline_model is None:
            raise ValueError("模型尚未训练，请先调用 train() 方法。")

        test_set, test_loader = self._get_data(flag='test')
        
        print("在测试集上生成基线预测、真实标签和输入窗口...")
        yhat_te_bl, y_te_synced, W_te_synced = self._get_predictions(self.baseline_model, test_loader)

        print("使用 Hybrid Corrector 对基线预测进行校正...")
        yfinal_te = self.hybrid_corrector.predict(W_te_synced, yhat_te_bl)

        # --- 评估和结果保存 ---
        print("\n" + "=" * 50)
        print("--- 最终性能评估 ---")
        print("=" * 50)

        # 计算基线模型的结果
        base_mae, base_mse, _, _, _ = metric(yhat_te_bl, y_te_synced)
        print(f"基线模型 ({self.args.baseline_model}) -> MSE: {base_mse:.4f}, MAE: {base_mae:.4f}")

        # 计算 CRC 校正后的结果
        final_mae, final_mse, _, _, _ = metric(yfinal_te, y_te_synced)
        print(f"CRC 校正后模型 -> MSE: {final_mse:.4f}, MAE: {final_mae:.4f}")
        
        # 8. 新增功能：可选的DTW评估，更好地衡量长序列的形状相似度
        if self.args.use_dtw:
            print("正在计算 DTW 距离...")
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(yfinal_te.shape[0]):
                # 假设我们只评估最后一个通道
                x = yfinal_te[i, :, -1].reshape(-1, 1)
                y = y_te_synced[i, :, -1].reshape(-1, 1)
                if i % 100 == 0:
                    print(f"  calculating dtw iter: {i}/{yfinal_te.shape[0]}")
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw_score = np.array(dtw_list).mean()
            print(f"CRC 校正后模型 -> DTW: {dtw_score:.4f}")
        
        print("-" * 50)
        mae_improvement = (base_mae - final_mae) / max(base_mae, 1e-8) * 100
        mse_improvement = (base_mse - final_mse) / max(base_mse, 1e-8) * 100
        print(f"MAE 相对提升率: {mae_improvement:.2f}%")
        print(f"MSE 相对提升率: {mse_improvement:.2f}%")
        print("=" * 50)

        # 添加画图功能，更加直观
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        print("正在生成可视化图片...")
        for i in range(min(W_te_synced.shape[0], 100)): # 限制最多生成100张图，防止过多
            if i % 20 == 0: # 每隔20个样本保存一张图
                gt = np.concatenate((W_te_synced[i, :, -1], y_te_synced[i, :, -1]), axis=0)
                pd = np.concatenate((W_te_synced[i, :, -1], yfinal_te[i, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        print(f"可视化图片已保存至: {folder_path}")
        
        # 保存结果的逻辑保持不变
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'metrics.npy', np.array([final_mae, final_mse]))
        np.save(folder_path + 'pred.npy', yfinal_te)
        np.save(folder_path + 'true.npy', y_te_synced)
        np.save(folder_path + 'pred_baseline.npy', yhat_te_bl)
        
        return