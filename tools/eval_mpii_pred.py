from collections import OrderedDict

import numpy as np
from scipy.io import loadmat


def origin_eval(
    gt_file='/home/zyq/code/mmpose/data/mpii/annotations/mpii_gt_val.mat',
    preds_file='/home/zyq/code/mmpose/work_dirs/hrnet_w32_mpii_256x256_mspn/pred.mat'
):
    SC_BIAS = 0.6
    threshold = 0.5
    gt_dict = loadmat(gt_file)
    preds = loadmat(preds_file)['preds']
    dataset_joints = gt_dict['dataset_joints']
    jnt_missing = gt_dict['jnt_missing']
    pos_gt_src = gt_dict['pos_gt_src']
    headboxes_src = gt_dict['headboxes_src']
    pos_pred_src = np.transpose(preds, [1, 2, 0])
    head = np.where(dataset_joints == 'head')[1][0]
    lsho = np.where(dataset_joints == 'lsho')[1][0]
    lelb = np.where(dataset_joints == 'lelb')[1][0]
    lwri = np.where(dataset_joints == 'lwri')[1][0]
    lhip = np.where(dataset_joints == 'lhip')[1][0]
    lkne = np.where(dataset_joints == 'lkne')[1][0]
    lank = np.where(dataset_joints == 'lank')[1][0]
    rsho = np.where(dataset_joints == 'rsho')[1][0]
    relb = np.where(dataset_joints == 'relb')[1][0]
    rwri = np.where(dataset_joints == 'rwri')[1][0]
    rkne = np.where(dataset_joints == 'rkne')[1][0]
    rank = np.where(dataset_joints == 'rank')[1][0]
    rhip = np.where(dataset_joints == 'rhip')[1][0]
    jnt_visible = 1 - jnt_missing
    uv_error = pos_pred_src - pos_gt_src
    uv_err = np.linalg.norm(uv_error, axis=1)
    headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
    headsizes = np.linalg.norm(headsizes, axis=0)
    headsizes *= SC_BIAS
    scale = headsizes * np.ones((len(uv_err), 1), dtype=np.float32)
    scaled_uv_err = uv_err / scale
    scaled_uv_err = scaled_uv_err * jnt_visible
    jnt_count = np.sum(jnt_visible, axis=1)
    less_than_threshold = (scaled_uv_err <= threshold) * jnt_visible
    PCKh = 100. * np.sum(less_than_threshold, axis=1) / jnt_count
    # save
    rng = np.arange(0, 0.5 + 0.01, 0.01)
    pckAll = np.zeros((len(rng), 16), dtype=np.float32)
    for r, threshold in enumerate(rng):
        less_than_threshold = (scaled_uv_err <= threshold) * jnt_visible
        pckAll[r, :] = 100. * np.sum(less_than_threshold, axis=1) / jnt_count
    PCKh = np.ma.array(PCKh, mask=False)
    PCKh.mask[6:8] = True
    jnt_count = np.ma.array(jnt_count, mask=False)
    jnt_count.mask[6:8] = True
    jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)
    name_value = [('Head', PCKh[head]),
                  ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
                  ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
                  ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
                  ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
                  ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
                  ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
                  ('PCKh', np.sum(PCKh * jnt_ratio)),
                  ('PCKh@0.1', np.sum(pckAll[11, :] * jnt_ratio))]
    name_value = OrderedDict(name_value)


def find_lower(
        gt_file='/home/zyq/code/mmpose/data/mpii/annotations/mpii_gt_val.mat',
        preds_file='/home/zyq/code/mmpose/work_dirs/hrnet_w32_mpii_256x256_mspn/pred.mat',
        threshold=0.11):
    SC_BIAS = 0.6
    gt_dict = loadmat(gt_file)
    preds = loadmat(preds_file)['preds']
    dataset_joints = gt_dict['dataset_joints']
    jnt_missing = gt_dict['jnt_missing']
    pos_gt_src = gt_dict['pos_gt_src']
    headboxes_src = gt_dict['headboxes_src']
    pos_pred_src = np.transpose(preds, [1, 2, 0])
    kpt_idx = {
        'head': np.where(dataset_joints == 'head')[1][0],
        'lsho': np.where(dataset_joints == 'lsho')[1][0],
        'lelb': np.where(dataset_joints == 'lelb')[1][0],
        'lwri': np.where(dataset_joints == 'lwri')[1][0],
        'lhip': np.where(dataset_joints == 'lhip')[1][0],
        'lkne': np.where(dataset_joints == 'lkne')[1][0],
        'lank': np.where(dataset_joints == 'lank')[1][0],
        'rsho': np.where(dataset_joints == 'rsho')[1][0],
        'relb': np.where(dataset_joints == 'relb')[1][0],
        'rwri': np.where(dataset_joints == 'rwri')[1][0],
        'rkne': np.where(dataset_joints == 'rkne')[1][0],
        'rank': np.where(dataset_joints == 'rank')[1][0],
        'rhip': np.where(dataset_joints == 'rhip')[1][0]
    }
    jnt_visible = 1 - jnt_missing
    uv_error = pos_pred_src - pos_gt_src
    uv_err = np.linalg.norm(uv_error, axis=1)
    headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
    headsizes = np.linalg.norm(headsizes, axis=0)
    headsizes *= SC_BIAS
    scale = headsizes * np.ones((len(uv_err), 1), dtype=np.float32)
    scaled_uv_err = uv_err / scale
    scaled_uv_err = scaled_uv_err * jnt_visible
    # save
    scaled_uv_err_ma = np.ma.masked_where(scaled_uv_err <= threshold,
                                          scaled_uv_err)
    mask = scaled_uv_err_ma.mask
    mask.sum(axis=0)


if __name__ == '__main__':
    find_lower()
