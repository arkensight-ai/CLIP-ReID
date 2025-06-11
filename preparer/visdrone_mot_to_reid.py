from collections import defaultdict
import os
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import glob
SPLITS = {
    "train": {
        "base_dir": "../VisDrone2019-MOT-train/sequences",
        "ann_dir": "../VisDrone2019-MOT-train/annotations",
        "output_dir": "./data/visdrone_mot_reid/bounding_boxes_train",
        "list_file": "train_list.txt"
    },
    "val": {
        "base_dir": "../VisDrone2019-MOT-val/sequences",
        "ann_dir": "../VisDrone2019-MOT-val/annotations",
        "output_dir": "./data/visdrone_mot_reid/bounding_boxes_val",
        "list_file": "test_list.txt"
    },
    "test": {
        "base_dir": "../VisDrone2019-MOT-test-dev/sequences",
        "ann_dir": "../VisDrone2019-MOT-test-dev/annotations",
        "output_dir": "./data/visdrone_mot_reid/bounding_boxes_test",
        "list_file": "test_list.txt"
    }
}

OUTPUT_DIR_ANN = "./data/visdrone_mot_reid/train_val_test"
OUTPUT_DIR = "../data/visdrone_mot_reid/bounding_boxes"

PADDING_PERCENTAGE = 0.30
PROCESS_IMAGES = True
debug = False

if not os.path.exists(OUTPUT_DIR_ANN):
    os.makedirs(OUTPUT_DIR_ANN, exist_ok=True)
for f in glob.glob(os.path.join(OUTPUT_DIR_ANN, "*.txt")):
    os.remove(f)

global_id_dict = defaultdict(lambda: len(global_id_dict))
process_bbox_jobs = []

def process_bbox(job):
    img_path, x, y, w, h, pad_pct, output_path = job
    image = cv2.imread(img_path)
    if image is None:
        return
    H, W, _ = image.shape
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    pad_x = pad_pct * (x2 - x1)
    pad_y = pad_pct * (y2 - y1)
    x1 = int(max(0, x1 - pad_x))
    x2 = int(min(W, x2 + pad_x))
    y1 = int(max(0, y1 - pad_y))
    y2 = int(min(H, y2 + pad_y))
    bbox = image[y1:y2, x1:x2]
    cv2.imwrite(output_path, bbox)

for split, cfg in SPLITS.items():
    base_dir = cfg["base_dir"]
    ann_dir = cfg["ann_dir"] if cfg["ann_dir"] is not None else base_dir.replace("sequences", "annotations")
    output_dir_split = cfg["output_dir"]
    list_file = os.path.join(OUTPUT_DIR_ANN, cfg["list_file"])

    if not os.path.exists(output_dir_split):
        os.makedirs(output_dir_split, exist_ok=True)

    for cam_id, sequence_name in enumerate(sorted(os.listdir(base_dir))):
        seq_path = os.path.join(base_dir, sequence_name)
        img_folder = seq_path
        output_dir = os.path.join(output_dir_split, sequence_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        if not os.path.isdir(seq_path):
            continue

        gt = os.path.join(ann_dir, f"{sequence_name}.txt")
        if not os.path.exists(gt):
            continue

        with open(gt, "r") as gt_file:
            gt_data = gt_file.readlines()

        for ground_truth in tqdm(gt_data, desc=f"{split}-{sequence_name}"):
            parts = ground_truth.strip().split(',')
            if len(parts) < 6:
                continue
            frame_id, target_id, x, y, w, h = parts[:6]
            x, y, w, h = int(float(x)), int(float(y)), int(float(w)), int(float(h))
            global_id = global_id_dict[(sequence_name, target_id)]
            img_name = f"{int(frame_id):07}.jpg"
            img_path = os.path.join(img_folder, img_name)
            filename = f"{global_id:04d}_c{cam_id}_f{int(frame_id):04d}.jpg"
            output_path = os.path.join(output_dir, filename)

            if debug:
                import pdb
                pdb.set_trace()
            
            process_bbox_jobs.append((img_path, x, y, w, h, PADDING_PERCENTAGE, output_path))
            with open(list_file, 'a') as f:
                f.write(f"{filename} {global_id} {sequence_name}\n")

if PROCESS_IMAGES:
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_bbox, process_bbox_jobs), total=len(process_bbox_jobs)))