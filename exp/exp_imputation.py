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
import contextlib

warnings.filterwarnings('ignore')


class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)
        self.data_source = getattr(self.args, 'data_source', 'tslib')

        # Precision setup
        precision = getattr(self.args, 'precision', 'bf16')
        dtype_map = {'fp32': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}
        self.precision_dtype = dtype_map.get(precision, None)

    def _autocast_ctx(self):
        if self.precision_dtype is not None and self.device.type == 'cuda':
            return torch.autocast(device_type='cuda', dtype=self.precision_dtype)
        return contextlib.nullcontext()

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            if self.data_source == 'tslib':
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)

                    B, T, N = batch_x.shape
                    mask = torch.rand((B, T, N)).to(self.device)
                    mask[mask <= self.args.mask_rate] = 0
                    mask[mask > self.args.mask_rate] = 1
                    inp = batch_x.masked_fill(mask == 0, 0)

                    with self._autocast_ctx():
                        outputs = self.model(inp, batch_x_mark, None, None, mask)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    batch_x = batch_x[:, :, f_dim:]
                    mask = mask[:, :, f_dim:]

                    loss = criterion(outputs.detach()[mask == 0], batch_x.detach()[mask == 0])
                    total_loss.append(loss.item())
            else:
                for i, (batch_x, batch_x_ori, batch_mask, batch_indicating, batch_x_mark, _) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_x_ori = batch_x_ori.float().to(self.device)
                    batch_mask = batch_mask.float().to(self.device)
                    batch_indicating = batch_indicating.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)

                    with self._autocast_ctx():
                        outputs = self.model(batch_x, batch_x_mark, None, None, batch_mask)

                    eval_mask = (1 - batch_mask)
                    if batch_indicating.sum() > 0:
                        eval_mask = batch_indicating

                    loss = criterion(outputs.detach()[eval_mask == 1], batch_x_ori.detach()[eval_mask == 1])
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

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            if self.data_source == 'tslib':
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()

                    batch_x = batch_x.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)

                    B, T, N = batch_x.shape
                    mask = torch.rand((B, T, N)).to(self.device)
                    mask[mask <= self.args.mask_rate] = 0
                    mask[mask > self.args.mask_rate] = 1
                    inp = batch_x.masked_fill(mask == 0, 0)

                    with self._autocast_ctx():
                        outputs = self.model(inp, batch_x_mark, None, None, mask)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, :, f_dim:]
                        batch_x = batch_x[:, :, f_dim:]
                        mask = mask[:, :, f_dim:]
                        loss = criterion(outputs[mask == 0], batch_x[mask == 0])

                    train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    loss.backward()
                    model_optim.step()
            else:
                for i, (batch_x, batch_x_ori, batch_mask, batch_indicating, batch_x_mark, _) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()

                    batch_x = batch_x.float().to(self.device)
                    batch_x_ori = batch_x_ori.float().to(self.device)
                    batch_mask = batch_mask.float().to(self.device)
                    batch_indicating = batch_indicating.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)

                    with self._autocast_ctx():
                        outputs = self.model(batch_x, batch_x_mark, None, None, batch_mask)
                        loss = criterion(outputs[batch_indicating == 1], batch_x_ori[batch_indicating == 1])

                    train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
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
        masks = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            if self.data_source == 'tslib':
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)

                    B, T, N = batch_x.shape
                    mask = torch.rand((B, T, N)).to(self.device)
                    mask[mask <= self.args.mask_rate] = 0
                    mask[mask > self.args.mask_rate] = 1
                    inp = batch_x.masked_fill(mask == 0, 0)

                    with self._autocast_ctx():
                        outputs = self.model(inp, batch_x_mark, None, None, mask)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    batch_x = batch_x[:, :, f_dim:]
                    mask = mask[:, :, f_dim:]

                    outputs = outputs.detach().cpu().numpy()
                    pred = outputs
                    true = batch_x.detach().cpu().numpy()
                    preds.append(pred)
                    trues.append(true)
                    masks.append(mask.detach().cpu())

                    if i % 20 == 0:
                        filled = true[0, :, -1].copy()
                        filled = filled * mask[0, :, -1].detach().cpu().numpy() + \
                                 pred[0, :, -1] * (1 - mask[0, :, -1].detach().cpu().numpy())
                        visual(true[0, :, -1], filled, os.path.join(folder_path, str(i) + '.pdf'))
            else:
                for i, (batch_x, batch_x_ori, batch_mask, batch_indicating, batch_x_mark, _) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_x_ori = batch_x_ori.float().to(self.device)
                    batch_mask = batch_mask.float().to(self.device)
                    batch_indicating = batch_indicating.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)

                    with self._autocast_ctx():
                        outputs = self.model(batch_x, batch_x_mark, None, None, batch_mask)

                    outputs = outputs.detach().cpu().numpy()
                    pred = outputs
                    true = batch_x_ori.detach().cpu().numpy()
                    eval_mask = batch_indicating.detach().cpu().numpy()
                    if eval_mask.sum() == 0:
                        eval_mask = (1 - batch_mask.detach().cpu().numpy())

                    preds.append(pred)
                    trues.append(true)
                    masks.append(eval_mask)

                    if i % 20 == 0:
                        filled = true[0, :, -1].copy()
                        m = eval_mask[0, :, -1]
                        filled = filled * (1 - m) + pred[0, :, -1] * m
                        visual(true[0, :, -1], filled, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if self.data_source == 'tslib':
            mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
        else:
            mae, mse, rmse, mape, mspe = metric(preds[masks == 1], trues[masks == 1])

        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_imputation.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return
