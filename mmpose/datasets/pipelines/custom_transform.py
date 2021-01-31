from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from ..registry import PIPELINES


def save_image_coco(results, output_path):
    link_pairs = [
        [15, 13],
        [13, 11],
        [11, 5],
        [12, 14],
        [14, 16],
        [12, 6],
        # [3, 1],[1, 2],[1, 0],[0, 2],[2,4],
        [9, 7],
        [7, 5],
        [5, 6],
        [6, 8],
        [8, 10],
    ]
    joints_to_color = [
        [210, 245, 60],  # nose yellow
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [230, 25, 75],
        [230, 25, 75],  # shoulder red
        [60, 180, 75],
        [60, 180, 75],  # elbow green
        [0, 130, 200],
        [0, 130, 200],  # wrist blue
        [245, 130, 48],
        [245, 130, 48],  # hip orange
        [70, 240, 240],
        [70, 240, 240],  # knee light blue
        [145, 30, 180],
        [145, 30, 180]  # ankle purple
    ]
    line_color = (0, 120, 212)
    image_file = Path(results['image_file'])
    if 'joints' in results:
        keypoints = results['joints'][0]
        if (keypoints.shape[0] > 1):
            print(image_file)
            return
        keypoints = np.round(keypoints[0]).astype(int).tolist()
        # print(keypoints)
    elif 'joints_3d' in results:
        keypoints = results['joints_3d']
        # print(keypoints)
    else:
        print(results.keys())
        return
    img = cv2.cvtColor(np.array(results['img']), cv2.COLOR_RGB2BGR)
    for idx, joint in enumerate(keypoints):
        if idx in (1, 2, 3, 4):
            continue
        cv2.circle(img, (joint[0], joint[1]), 5,
                   list(reversed(joints_to_color[idx])), -1)
    for line in link_pairs:
        cv2.line(img, tuple(keypoints[line[0]][:2]),
                 tuple(keypoints[line[1]][:2]), line_color, 2)
    image_name = image_file.stem
    image_name = '-'.join(
        list(image_file.parts[-3:-1]) + [image_name + '.jpg'])
    cv2.imwrite(str(output_path / image_name), img)
    return results


def save_image_mpii(results, output_path):
    link_pairs = [(0, 1), (1, 2), (2, 6), (7, 12), (12, 11), (11, 10), (5, 4),
                  (4, 3), (3, 6), (7, 13), (13, 14), (14, 15), (6, 7), (7, 8),
                  (8, 9)]
    image_file = Path(results['image_file'])
    if 'joints' in results:
        keypoints = results['joints'][0]
        if (keypoints.shape[0] > 1):
            print(image_file)
            return
        keypoints = np.round(keypoints[0]).astype(int).tolist()
        # print(keypoints)
    elif 'joints_3d' in results:
        keypoints = results['joints_3d']
        # print(keypoints)
    else:
        print(results.keys())
        return
    img = cv2.cvtColor(np.array(results['img']), cv2.COLOR_RGB2BGR)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(link_pairs) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(link_pairs)):
        i1 = link_pairs[l][0]
        i2 = link_pairs[l][1]
        p1 = keypoints[i1, 0].astype(np.int32), keypoints[i1,
                                                          1].astype(np.int32)
        p2 = keypoints[i2, 0].astype(np.int32), keypoints[i2,
                                                          1].astype(np.int32)
        cv2.line(
            kp_mask,
            p1,
            p2,
            color=colors[l],
            thickness=2,
            lineType=cv2.LINE_AA)
        cv2.circle(
            kp_mask,
            p1,
            radius=3,
            color=colors[l],
            thickness=-1,
            lineType=cv2.LINE_AA)
        cv2.circle(
            kp_mask,
            p2,
            radius=3,
            color=colors[l],
            thickness=-1,
            lineType=cv2.LINE_AA)
    image_name = image_file.stem
    image_name = '-'.join(
        list(image_file.parts[-3:-1]) + [image_name + '.jpg'])
    cv2.imwrite(str(output_path / image_name), kp_mask)
    return results


@PIPELINES.register_module()
class SaveImageRandom:
    """Save input image with keypoint.

    Required key: 'img'

    Args:
        results (dict): contain all information about training.
    """

    def __init__(self,
                 p=0.5,
                 output_path='visualization/train/',
                 dataset='coco'):
        self.p = p
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        if dataset == 'coco':
            self.save_image = save_image_coco
        elif dataset == 'mpii':
            self.save_image = save_image_mpii
        else:
            raise NotImplementedError

    def __call__(self, results):
        if np.random.rand() <= self.p:
            self.save_image(results, self.output_path)
        return results


@PIPELINES.register_module()
class TopDownRandomShiftBBox():

    def _xywh2cs(self, x, y, w, h, image_size):
        """This encodes bbox(x,y,w,w) into (center, scale)

        Args:
            x, y, w, h

        Returns:
            tuple: A tuple containing center and scale.

            - center (np.ndarray[float32](2,)): center of the bbox (x, y).
            - scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """
        aspect_ratio = image_size[0] / image_size[1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        if np.random.rand() < 0.3:
            center += 0.4 * (np.random.rand(2) - 0.5) * [w, h]

        # if w > aspect_ratio * h:
        #     h = w * 1.0 / aspect_ratio
        # elif w < aspect_ratio * h:
        #     w = h * aspect_ratio

        # pixel std is 200.0
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)

        return center, scale

    def __call__(self, results):
        """Perform data augmentation with random enlarge bounding box."""

        shift = np.random.rand(2) * 0.3  # [0.0, 0.3]
        if 'bbox' in results:
            x, y, w, h = results['bbox']
        else:
            center = results['center']
            scale = results['scale']
            x = max(0, center[0] - scale[0] * 200 / 2)
            y = max(0, center[1] - scale[0] * 200 / 2)
            w = h = scale[0] * 200
        image_size = results['ann_info']['image_size']
        w, h = (1 + shift) * [w, h]
        x, y = np.array([x, y]) - shift * [w, h] / 2
        x = max(0.0, x)
        y = max(0.0, y)
        center, scale = self._xywh2cs(x, y, w, h, image_size)
        results['center'] = center
        results['scale'] = scale
        results['bbox'] = [x, y, w, h]
        return results
