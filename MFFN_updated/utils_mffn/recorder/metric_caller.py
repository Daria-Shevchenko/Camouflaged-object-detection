# -*- coding: utf-8 -*-
# @Time    : 2020/12/19
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import numpy as np
from py_sod_metrics.sod_metrics import Emeasure, Fmeasure, MAE, Smeasure, WeightedFmeasure



class IoU(object):
    def __init__(self, threshold=128, eps=1e-8):
        self.threshold = threshold
        self.eps = eps
        self.ious = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred_bin = pred >= self.threshold
        gt_bin = gt >= 128

        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()

        iou = intersection / (union + self.eps)
        self.ious.append(iou)

    def get_results(self):
        return {"iou": float(np.mean(self.ious)) if len(self.ious) > 0 else 0.0}

class CalTotalMetric(object):
    __slots__ = ["cal_mae", "cal_fm", "cal_sm", "cal_em", "cal_wfm", "cal_iou"]

    def __init__(self):
        self.cal_mae = MAE()
        self.cal_fm = Fmeasure()
        self.cal_sm = Smeasure()
        self.cal_em = Emeasure()
        self.cal_wfm = WeightedFmeasure()
        self.cal_iou = IoU()

    def step(self, pred: np.ndarray, gt: np.ndarray, gt_path: str):
        assert pred.ndim == gt.ndim and pred.shape == gt.shape, (pred.shape, gt.shape, gt_path)
        assert pred.dtype == np.uint8, pred.dtype
        assert gt.dtype == np.uint8, gt.dtype

        self.cal_mae.step(pred, gt)
        self.cal_fm.step(pred, gt)
        self.cal_sm.step(pred, gt)
        self.cal_em.step(pred, gt)
        self.cal_wfm.step(pred, gt)
        self.cal_iou.step(pred, gt)

    def get_results(self, bit_width: int = 3) -> dict:
        fm = self.cal_fm.get_results()["fm"]
        wfm = self.cal_wfm.get_results()["wfm"]
        sm = self.cal_sm.get_results()["sm"]
        em = self.cal_em.get_results()["em"]
        mae = self.cal_mae.get_results()["mae"]
        iou = self.cal_iou.get_results()["iou"]
        results = {
            "Smeasure": sm,
            "wFmeasure": wfm,
            "MAE": mae,
            "adpEm": em["adp"],
            "meanEm": em["curve"].mean(),
            "maxEm": em["curve"].max(),
            "adpFm": fm["adp"],
            "meanFm": fm["curve"].mean(),
            "maxFm": fm["curve"].max(),
            "IoU": iou,
        }

        def _round_w_zero_padding(_x):
            # _x = str(_x.round(bit_width))
            _x = str(round(_x, bit_width))
            _x += "0" * (bit_width - len(_x.split(".")[-1]))
            return _x

        results = {name: _round_w_zero_padding(metric) for name, metric in results.items()}
        return results
