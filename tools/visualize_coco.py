import os

import cv2
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from xtcocotools.coco import COCO

if __name__ == '__main__':
    coco = COCO('data/keypoint_merged_train.json')
    os.makedirs('visualization/coco/', exist_ok=True)
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
    for ann_id in coco.getAnnIds():
        anno = coco.loadAnns(ann_id)[0]
        image_info = coco.loadImgs(int(anno['image_id']))[0]
        keypoints = anno['keypoints']
        keypoints = [[
            int(keypoints[i]),
            int(keypoints[i + 1]), keypoints[i + 2]
        ] for i in range(0, len(keypoints), 3)]
        img = cv2.imread(str(image_info['file_name']))
        for idx, joint in enumerate(keypoints):
            if idx in (1, 2, 3, 4):
                continue
            cv2.circle(img, (joint[0], joint[1]), 5,
                       list(reversed(joints_to_color[idx])), -1)
        for line in link_pairs:
            cv2.line(img, tuple(keypoints[line[0]][:2]),
                     tuple(keypoints[line[1]][:2]), line_color, 2)
        cv2.imwrite(f'visualization/coco/{ann_id}.png', img)
