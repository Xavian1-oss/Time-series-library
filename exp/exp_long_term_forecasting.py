from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')

# >>> PIR integration: imports
try:
    from models.pir.backbone_adapter import BackboneAdapter
    from models.pir.pir_wrapper import PIRWrapper
    _HAS_PIR = True
except Exception:
    _HAS_PIR = False


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        # >>> PIR integration: 先初始化 PIR 相关属性，避免 _build_model() 访问时报 AttributeError
        self.pir = None
        # 尽量只用 args 取值，提前缓存为成员，_build_model() 会用到
        self._use_pir = getattr(args, 'use_pir', 0)
        self.pir_target_channel = getattr(args, 'pir_target_channel', 0)
        self._pir_topk = getattr(args, 'pir_topk', 20)
        self._pir_lambda_aux = getattr(args, 'pir_lambda_aux', 1.0)

        # 调用父类构造函数（这里面会触发 _build_model()）
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # >>> PIR integration: wrap backbone & build PIR when enabled
        use_pir = getattr(self, '_use_pir', getattr(self.args, 'use_pir', 0))
        if use_pir and _HAS_PIR:
            N = self.args.enc_in
            Lin, Lout = self.args.seq_len, self.args.pred_len
            exo_dim = 4  # 占位；如需更强，可用真实 time features
            self.backbone_adapter = BackboneAdapter(model, Lin, Lout, N)
            self.pir = PIRWrapper(
                backbone=self.backbone_adapter,
                n_channels=N,
                lin=Lin,
                lout=Lout,
                exo_dim=exo_dim,
                topk=getattr(self, '_pir_topk', getattr(self.args, 'pir_topk', 20)),
                lambda_aux=getattr(self, '_pir_lambda_aux', getattr(self.args, 'pir_lambda_aux', 1.0)),
                target_channel=getattr(self, 'pir_target_channel', getattr(self.args, 'pir_target_channel', 0)),
            ).to(self.device)
        else:
            self.backbone_adapter = None

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # >>> PIR integration: 若使用 PIR，用 PIR 的参数（含backbone）做优化
        if self.pir is not None:
            return optim.Adam(self.pir.parameters(), lr=self.args.learning_rate)
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    # >>> PIR integration: 小工具，把 (B, L, N) 转 (B, N, L)
    @staticmethod
    def _to_BNL(x_LN):
        # x_LN: (B, L, N)
        return x_LN.transpose(1, 2).contiguous()

    def _build_pir_retrieval(self, loader, max_batches=200):
        Xs, Ys = [], []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                xb = batch_x.float().to(self.device)                 # (B, Lin, N)
                yb = batch_y.float().to(self.device)                 # (B, label_len+pred_len, N)
                yb = yb[:, -self.args.pred_len:, :]                  # >>> 只保留 pred_len 段 (B, pred_len, N)
                Xs.append(self._to_BNL(xb))                          # -> (B, N, Lin)
                Ys.append(self._to_BNL(yb))                          # -> (B, N, pred_len)
                if i + 1 >= max_batches:
                    break
        Xref = torch.cat(Xs, dim=0)
        Yref = torch.cat(Ys, dim=0)
        self.pir.build_retrieval(Xref, Yref)


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)        # (B, Lin, N)
                batch_y = batch_y.float().to(self.device)        # (B, Lout, N)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                use_pir = getattr(self.args, 'use_pir', 0) and (self.pir is not None)
                if use_pir:
                    # 用 PIR：x->(B,N,Lin), y->(B,N,Lout), exo 占位
                    x_bnL = self._to_BNL(batch_x)
                    batch_y = batch_y[:, -self.args.pred_len:, :]           # (B, pred_len, N)
                    y_bnL = self._to_BNL(batch_y)
                    exo = torch.zeros(x_bnL.size(0), 4, device=x_bnL.device)
                    out = self.pir(x_bnL, exo, y_bnL)   # 包含 y_bar & y_pred_t
                    # 组装 outputs：(B, Lout, N)
                    y_bar_LN = out["y_bar"].transpose(1, 2).contiguous()  # (B,Lout,N)
                    # 替换目标通道为融合后的预测
                    y_bar_LN[:, :, self.pir_target_channel] = out["y_pred_t"]
                    outputs = y_bar_LN  # (B, Lout, N)

                else:
                    # 原路径
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # 通道维切片（两条路径统一在这里做）
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]            # (B, pred_len, C')
                batch_y = batch_y[:, :, f_dim:]            # (B, pred_len, C')

                pred = outputs.detach()
                true = batch_y.detach()
                loss = criterion(pred, true)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # >>> PIR integration: 在第一轮前构建检索库
        if getattr(self.args, 'use_pir', 0) and (self.pir is not None):
            self._build_pir_retrieval(train_loader, max_batches=200)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            # 注意：即使用 PIR，backbone 仍需要 train() 以启用 Dropout/BN 等
            self.model.train()
            if self.pir is not None:
                self.pir.train()

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)       # (B, Lin, N)
                batch_y = batch_y.float().to(self.device)       # (B, Lout, N)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                use_pir = getattr(self.args, 'use_pir', 0) and (self.pir is not None)
                if use_pir:
                    # PIR 路径
                    x_bnL = self._to_BNL(batch_x)
                    y_pred_part = batch_y[:, -self.args.pred_len:, :]
                    y_bnL = self._to_BNL(y_pred_part)
                    exo = torch.zeros(x_bnL.size(0), 4, device=x_bnL.device)

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            out = self.pir(x_bnL, exo, y_bnL)
                            loss = out["loss"]
                            train_loss.append(loss.item())
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        out = self.pir(x_bnL, exo, y_bnL)
                        loss = out["loss"]
                        train_loss.append(loss.item())
                        loss.backward()
                        model_optim.step()
                else:
                    # 原路径
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                        loss.backward()
                        model_optim.step()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # 这里的 early_stopping 仍然保存的是 backbone 的权重（与原版一致）
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        if self.pir is not None:
            self.pir.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)        # (B, Lin, N)
                batch_y = batch_y.float().to(self.device)        # (B, Lout, N)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                use_pir = getattr(self.args, 'use_pir', 0) and (self.pir is not None)
                if use_pir:
                    x_bnL = self._to_BNL(batch_x)
                    exo = torch.zeros(x_bnL.size(0), 4, device=x_bnL.device)
                    # 推理时不需要 y
                    out = self.pir(x_bnL, exo, y=None)
                    y_bar_LN = out["y_bar"].transpose(1, 2).contiguous()  # (B, Lout, N)
                    y_bar_LN[:, :, self.pir_target_channel] = out["y_pred_t"]
                    outputs = y_bar_LN  # (B, Lout, N)
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, :]

                # 反归一化与可视化均沿用原逻辑
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y_cut = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs_np = outputs.detach().cpu().numpy()
                batch_y_np = batch_y_cut.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y_np.shape
                    if outputs_np.shape[-1] != batch_y_np.shape[-1]:
                        outputs_np = np.tile(outputs_np, [1, 1, int(batch_y_np.shape[-1] / outputs_np.shape[-1])])
                    outputs_np = test_data.inverse_transform(outputs_np.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y_np = test_data.inverse_transform(batch_y_np.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs_np = outputs_np[:, :, f_dim:]
                batch_y_np = batch_y_np[:, :, f_dim:]

                pred = outputs_np
                true = batch_y_np

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input_np = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input_np.shape
                        input_np = test_data.inverse_transform(input_np.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input_np[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input_np[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw_val = np.array(dtw_list).mean()
        else:
            dtw_val = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw_val))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw_val))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
