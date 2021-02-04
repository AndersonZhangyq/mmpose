import argparse
import os
import os.path as osp
from copy import deepcopy

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmpose.apis import (inference_top_down_pose_model, multi_gpu_test,
                         single_gpu_test, vis_pose_result)
from mmpose.core import wrap_fp16_model
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet


def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--visualize', action='store_true', help='visualize result')
    parser.add_argument(
        '--visualize_output_path',
        default='output/viz/',
        help='visualize result file path')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        default='mAP',
        nargs='+',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "mAP" for MSCOCO')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    args.work_dir = osp.join('./work_dirs',
                             osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # collect gt
    cfg.data.test.pipeline[-1]['meta_keys'].extend(['origin_joints_3d', 'origin_joints_3d_visible'])
    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=distributed,
        shuffle=False,
        drop_last=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    # build the model and load checkpoint
    model = build_posenet(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    _ = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # for backward compatibility
    model.to(torch.device('cuda:0'))
    model.cfg = cfg

    model.eval()
    results = []
    dataset = data_loader.dataset
    if args.visualize:
        prog_bar = mmcv.ProgressBar(len(dataset))
        for data in data_loader:
            img_metas = data['img_metas'].data[0][0]
            image_name = img_metas['image_file']

            # get gt
            gt_pose = img_metas['origin_joints_3d']
            gt_pose_vis = img_metas['origin_joints_3d_visible'][:, 0]
            gt_pose[:, -1] = gt_pose_vis

            # continue
            center = np.array(img_metas['center'], dtype=np.float)
            scale = np.array(img_metas['scale'], dtype=np.float)

            # Adjust center/scale slightly to avoid cropping limbs
            if center[0] != -1:
                center[1] = center[1] + 15 * scale[1]
                scale = scale * 1.25

            x = max(0, center[0] - scale[0] * 200 / 2)
            y = max(0, center[1] - scale[0] * 200 / 2)
            w = h = scale[0] * 200
            person_results = [{'bbox': [x, y, w, h]}]
            pose_results, returned_outputs = inference_top_down_pose_model(
                model,
                image_name,
                person_results,
                bbox_thr=None,
                format='xywh',
                dataset=cfg.data['test']['type'],
                return_heatmap=False,
                outputs=None)
            # add gt to result, only useable in single person case
            gt_result = deepcopy(pose_results[0])
            gt_result['keypoints'] = gt_pose

            pose_results.append(gt_result)

            vis_pose_result(
                model,
                image_name,
                pose_results,
                dataset=cfg.data['test']['type'],
                kpt_score_thr=0,
                show=False,
                out_file=osp.join(args.visualize_output_path,
                                  osp.basename(args.config[:-3]),
                                  osp.basename(image_name)))
            prog_bar.update()

    # for backward compatibility

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    eval_config = cfg.get('eval_config', {})
    eval_config = merge_configs(eval_config, dict(metric=args.eval))

    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)

        print(dataset.evaluate(outputs, args.work_dir, **eval_config))


if __name__ == '__main__':
    main()
