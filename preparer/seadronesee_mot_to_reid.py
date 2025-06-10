from collections import defaultdict
import os
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import json

BASE_DIR = '/home/markoharalovic/SeaDronesSee_MOT'
OUTPUT_DIR_TRAIN = '/home/markoharalovic/CLIP-ReID/data/SeaDronesSee_MOT/bounding_box_train'
OUTPUT_DIR_ANN_TRAIN = '/home/markoharalovic/CLIP-ReID/data/SeaDronesSee_MOT/train_test_split'

PADDING_PERCENTAGE = 0.30
PROCESS_IMAGES = True
debug = False

SPLITS = {
    'train': {
        'images': os.path.join(BASE_DIR, 'images', 'train'),
        'annotations': os.path.join(BASE_DIR, 'annotations', 'instances_train_objects_in_water.json')
    },
    'val' : {
        'images': os.path.join(BASE_DIR, 'images', 'val'),
        'annotations': os.path.join(BASE_DIR, 'annotations', 'instances_val_objects_in_water.json')
    }
}

os.makedirs(OUTPUT_DIR_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_DIR_ANN_TRAIN, exist_ok=True)

global_id_dict = defaultdict(lambda: len(global_id_dict))

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

process_bbox_jobs = []

for split in SPLITS:
    images_dir = SPLITS[split]['images']
    annotations_file = SPLITS[split]['annotations']

    print(f"Processing split: {split}")
    with open(annotations_file, 'r') as f:
        coco = json.load(f)

    image_id_to_info = {img["id"]: img for img in coco["images"]}

    output_txt_ann_file = f"{split}_list.txt"
    output_txt_ann_path = os.path.join(OUTPUT_DIR_ANN_TRAIN, output_txt_ann_file)

    for ann in tqdm(coco["annotations"]):
        image_info = image_id_to_info[ann["image_id"]]
        image_name = image_info["file_name"].replace(".png", ".jpg")
        image_path = os.path.join(images_dir, image_name)

        if debug: 
            import pdb
            pdb.set_trace()

        x, y, w, h = map(int, ann["bbox"])

        track_id = ann.get("track_id", ann.get("id")) 
        global_id = global_id_dict[(track_id, ann["category_id"] if "id" in ann else ann["image_id"])]

        cam_id = image_info.get("video_id", 0)
        frame_id = image_info.get("frame_index", image_info["id"])

        filename = f"{global_id:04d}_c{cam_id}_f{int(frame_id):04d}.jpg"
        sequence_name = image_info.get("source", {}).get("folder_name", "unknown_sequence")
        output_dir = os.path.join(OUTPUT_DIR_TRAIN, sequence_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        process_bbox_jobs.append((image_path, x, y, w, h, PADDING_PERCENTAGE, output_path))

        with open(output_txt_ann_path, 'a') as f:
            f.write(f"{filename} {global_id} {sequence_name}\n")

if PROCESS_IMAGES:
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_bbox, process_bbox_jobs), total=len(process_bbox_jobs)))