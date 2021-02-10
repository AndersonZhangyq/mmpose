import os, json, random
from itertools import filterfalse
from copy import deepcopy

from xtcocotools.coco import COCO


def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


if __name__ == "__main__":
    seed = 42
    output_root = "data/coco/annotations"
    anno_files = [
        "data/coco/annotations/person_keypoints_train2017.json",
        "data/coco/annotations/person_keypoints_train2017.json",
        "data/coco/annotations/person_keypoints_train2017.json",
        "data/coco/annotations/person_keypoints_train2017.json",
    ]
    percentages = [
        0.15,  # 42 hr similar size of mpii
        0.1,   # 28 hr
        0.05,  # 14 hr
        0.03,  # 8.4 hr
        0.01,  # 2.8 hr
    ]

    for json_file, percentage in zip(anno_files, percentages):
        coco_origin = COCO(json_file)
        anns_origin = []
        for item in list(coco_origin.anns.values()):
            if 'keypoints' not in item:
                continue
            if max(item['keypoints']) == 0:
                continue
            if 'num_keypoints' in item and item['num_keypoints'] == 0:
                continue
            anns_origin.append(item)
        origin_size = len(anns_origin)
        random.Random(seed).shuffle(anns_origin)
        anns_small = anns_origin[:int(origin_size * percentage)]
        anns_small = sorted(anns_small, key=lambda x: x['id'])
        images_small_ids = sorted(
            set([item['image_id'] for item in anns_small]))
        images_small = [coco_origin.imgs[id] for id in images_small_ids]
        dataset_small = coco_origin.dataset
        dataset_small['annotations'] = anns_small
        dataset_small['images'] = images_small
        print(f"selected annotation size {len(anns_small)}")
        print(f"selected images size {len(images_small)}")
        print(
            f"saving to {os.path.join(output_root, '{}_{}.json'.format(os.path.basename(json_file), percentage))}"
        )
        with open(
                os.path.join(
                    output_root,
                    f"{os.path.basename(json_file)[:-5]}_{percentage}.json"),
                "w+") as f:
            json.dump(dataset_small, f, ensure_ascii=False)
