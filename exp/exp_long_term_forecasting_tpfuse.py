
import os
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate

from models import TPFuse

warnings.filterwarnings("ignore")


class Exp_Long_Term_Forecast_TPFuse(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_TPFuse, self).__init__(args)

    def _build_model(self):
        model = TPFuse.Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate
        )

    def _select_criterion(self):
        return nn.MSELoss()

    @staticmethod
    def _safe_item(x, default=0.0):
        if x is None:
            return float(default)
        if isinstance(x, (int, float)):
            return float(x)
        if torch.is_tensor(x) and x.numel() > 0:
            return float(x.detach().float().cpu().item())
        return float(default)

    def _model_module(self):
        return self.model.module if isinstance(
            self.model, nn.DataParallel
        ) else self.model


    def _make_decoder_input(self, batch_y):
        zeros = torch.zeros_like(
            batch_y[:, -self.args.pred_len:, :]
        ).float()
        dec_inp = torch.cat(
            [batch_y[:, :self.args.label_len, :], zeros],
            dim=1
        )
        return dec_inp.float().to(self.device)

    def _forecast_one_batch(
        self,
        batch_x,
        batch_y,
        batch_x_mark,
        batch_y_mark,
        with_loss=False
    ):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = (
            None if batch_x_mark is None
            else batch_x_mark.float().to(self.device)
        )
        batch_y_mark = (
            None if batch_y_mark is None
            else batch_y_mark.float().to(self.device)
        )
        dec_inp = self._make_decoder_input(batch_y)

        if with_loss:
            forecast, loss_dict = self.model(
                batch_x,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                y_true=batch_y
            )
        else:
            forecast = self.model(
                batch_x,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                y_true=None
            )
            loss_dict = None

        f_dim = -1 if self.args.features == "MS" else 0
        forecast = forecast[:, -self.args.pred_len:, f_dim:]
        target = batch_y[:, -self.args.pred_len:, f_dim:]

        return forecast, target, loss_dict

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        squared_error = 0.0
        count = 0

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                forecast, target, _ = self._forecast_one_batch(
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    with_loss=False
                )
                diff = forecast.float() - target.float()
                squared_error += float(diff.pow(2).sum().cpu())
                count += diff.numel()

        self.model.train()
        return squared_error / max(count, 1)

    def train(self, setting):
        train_data, train_loader = self._get_data("train")
        vali_data, vali_loader = self._get_data("val")
        test_data, test_loader = self._get_data("test")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True
        )
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        use_amp = bool(getattr(self.args, "use_amp", False))
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        time_now = time.time()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_time = time.time()
            interval_start = time.time()

            train_sq = 0.0
            train_count = 0
            meters = {
                "total": [],
                "pred": [],
                "point": [],
                "check": [],
                "cons": [],
                "path_update": [],
                "correction": [],
                "gate": [],
                "gain": [],
            }
            interval = {k: [] for k in meters}

            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark
            ) in enumerate(train_loader):
                model_optim.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    forecast, target, loss_dict = self._forecast_one_batch(
                        batch_x,
                        batch_y,
                        batch_x_mark,
                        batch_y_mark,
                        with_loss=True
                    )
                    loss = loss_dict["L_total"]

                scaler.scale(loss).backward()
                scaler.unscale_(model_optim)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )
                scaler.step(model_optim)
                scaler.update()

                diff = forecast.detach().float() - target.detach().float()
                train_sq += float(diff.pow(2).sum().cpu())
                train_count += diff.numel()

                values = {
                    "total": self._safe_item(loss_dict["L_total"]),
                    "pred": self._safe_item(loss_dict["L_pred"]),
                    "point": self._safe_item(loss_dict["L_point"]),
                    "check": self._safe_item(loss_dict["L_check"]),
                    "cons": self._safe_item(loss_dict["L_consistency"]),
                    "path_update": self._safe_item(
                        loss_dict["residual_energy"]
                    ),
                    "correction": self._safe_item(
                        loss_dict["correction_energy"]
                    ),
                    "gate": self._safe_item(
                        loss_dict["mean_gate"]
                    ),
                    "gain": self._safe_item(loss_dict["decode_gain"]),
                }
                for key, value in values.items():
                    meters[key].append(value)
                    interval[key].append(value)

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - interval_start
                    speed = elapsed / 100.0
                    remaining = (
                        (self.args.train_epochs - epoch - 1)
                        * train_steps
                        + train_steps
                        - i
                        - 1
                    )
                    print(
                        "\titers: {0}, epoch: {1} | "
                        "total: {2:.7f} pred: {3:.7f} "
                        "point: {4:.7f} check: {5:.7f} "
                        "path_update: {6:.7f} correction: {7:.7f} "
                        "gate/gain: {8:.3f}/{9:.3f}".format(
                            i + 1,
                            epoch + 1,
                            np.mean(interval["total"]),
                            np.mean(interval["pred"]),
                            np.mean(interval["point"]),
                            np.mean(interval["check"]),
                            np.mean(interval["path_update"]),
                            np.mean(interval["correction"]),
                            np.mean(interval["gate"]),
                            np.mean(interval["gain"]),
                        )
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.1f}s".format(
                            speed,
                            speed * remaining
                        )
                    )
                    interval = {k: [] for k in meters}
                    interval_start = time.time()
                    time_now = time.time()

            train_mse = train_sq / max(train_count, 1)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            msg = (
                "Epoch: {0}, Steps: {1} | "
                "Train Loss: {2:.7f} Vali Loss: {3:.7f} "
                "Test Loss: {4:.7f} | "
                "Avg total: {5:.7f} pred: {6:.7f} "
                "point: {7:.7f} check: {8:.7f} "
                "cons: {9:.7f} path_update: {10:.7f} "
                "correction: {11:.7f} gate/gain: {12:.3f}/{13:.3f}"
            ).format(
                epoch + 1,
                train_steps,
                train_mse,
                vali_loss,
                test_loss,
                np.mean(meters["total"]),
                np.mean(meters["pred"]),
                np.mean(meters["point"]),
                np.mean(meters["check"]),
                np.mean(meters["cons"]),
                np.mean(meters["path_update"]),
                np.mean(meters["correction"]),
                np.mean(meters["gate"]),
                np.mean(meters["gain"]),
            )
            print("Epoch: {} cost time: {:.3f}s".format(
                epoch + 1,
                time.time() - epoch_time
            ))
            print(msg)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(
            torch.load(best_model_path, map_location=self.device)
        )
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data("test")

        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(
                    os.path.join(
                        self.args.checkpoints,
                        setting,
                        "checkpoint.pth"
                    ),
                    map_location=self.device
                )
            )

        sq_sum = 0.0
        abs_sum = 0.0
        ape_sum = 0.0
        spe_sum = 0.0
        count = 0
        eps = 1e-5


        self.model.eval()
        with torch.no_grad():
            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark
            ) in enumerate(test_loader):
                forecast, target, _ = self._forecast_one_batch(
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    with_loss=False
                )

                diff = forecast.float() - target.float()
                sq_sum += float(diff.pow(2).sum().cpu())
                abs_sum += float(diff.abs().sum().cpu())
                denom = target.float().abs().clamp_min(eps)
                ape_sum += float((diff.abs() / denom).sum().cpu())
                spe_sum += float((diff.pow(2) / denom.pow(2)).sum().cpu())
                count += diff.numel()


        mse = sq_sum / max(count, 1)
        mae = abs_sum / max(count, 1)
        rmse = float(np.sqrt(mse))
        mape = ape_sum / max(count, 1)
        mspe = spe_sum / max(count, 1)

        print("mse:{}, mae:{}".format(mse, mae))

        result_path = os.path.join("./results", setting)
        os.makedirs(result_path, exist_ok=True)
        np.save(
            os.path.join(result_path, "metrics.npy"),
            np.array([mae, mse, rmse, mape, mspe])
        )
        print("streaming metrics enabled; pred.npy and true.npy not saved")



