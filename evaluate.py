"""Geometry evaluation: symmetric normalized Chamfer and relaxed occupancy."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation


def read_mask(path):
    a=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise FileNotFoundError(path)
    return a < 128


def evaluate(pred: np.ndarray, gt: np.ndarray) -> dict:
    if pred.shape != gt.shape:
        raise ValueError('Prediction and ground truth must have the same size')
    if not np.any(pred) or not np.any(gt):
        return {'chamfer':float('inf'),'occupancy_percent':0.}
    dt_gt=distance_transform_edt(~gt)
    dt_pred=distance_transform_edt(~pred)
    diag=float(np.hypot(*pred.shape))
    chamfer=(float(np.mean(dt_gt[pred]))+float(np.mean(dt_pred[gt])))/(2.*diag)
    dilated=binary_dilation(gt,structure=np.ones((3,3),bool))
    occupancy=100.*float(np.count_nonzero(pred & dilated))/float(np.count_nonzero(pred))
    return {'chamfer':chamfer,'occupancy_percent':occupancy}


def main():
    p=argparse.ArgumentParser(description='Compare predicted centerline with ground truth')
    p.add_argument('prediction')
    p.add_argument('ground_truth')
    args=p.parse_args()
    print(json.dumps(evaluate(read_mask(args.prediction),read_mask(args.ground_truth)),indent=2))


if __name__=='__main__':main()
