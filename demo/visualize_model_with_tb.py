import mmcv
import torch
from torch.utils.tensorboard import SummaryWriter

from mmpose.datasets import build_dataset
from mmpose.models import build_posenet

if __name__ == '__main__':
    # config = "configs/videoeeg/mspn50_coco_256x192.py"
    # config = "configs/videoeeg/hrnet_w48_coco_384x288_dark_merged.py"
    # config = "configs/videoeeg/mspn50_coco_256x192_hrnet.py"
    # config = "configs/videoeeg/hrnet_w32_coco_256x192_udp.py"
    # config = "configs/paper/mpii/hrnet_w32_mpii_256x256_mspn.py"
    config = 'configs/top_down/mspn/coco/mspn50_coco_256x192.py'
    writer = SummaryWriter(f"viz/{config.split('/')[-1][:-3]}")
    config = mmcv.Config.fromfile(config)
    config.model.pretrained = None
    model = build_posenet(config.model)
    images = torch.randn([4, 3, 256, 192])
    # onnx.export does not support kwargs
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'Please implement the forward method for exporting.')
    # model.forward(images)
    writer.add_graph(model, images)
    writer.close()
