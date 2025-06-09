from collections import defaultdict
import os
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

BASE_DIR = '/home/markoharalovic/CLIP-ReID/data/VisDrone2019-MOT-test-dev/sequences'
OUTPUT_DIR_TRAIN = '/home/markoharalovic/CLIP-ReID/data/VisDrone2019-MOT-test-dev/bounding_box_train'
OUTPUT_DIR_ANN_TRAIN = '/home/markoharalovic/CLIP-ReID/data/VisDrone2019-MOT-test-dev/train_test_split'

TRAIN_SEQUENCES = [
    "uav0000073_04464_v",  "uav0000120_04775_v",  "uav0000201_00000_v",  "uav0000249_02688_v",  "uav0000297_02761_v",
    "uav0000370_00001_v", "uav0000119_02301_v"
]

TEST_SEQUENCES = [
     "uav0000188_00000_v",  "uav0000249_00001_v",  "uav0000297_00000_v",  "uav0000306_00230_v"
]

PADDING_PERCENTAGE = 0.30
PROCESS_IMAGES = False
debug = False

if not os.path.exists(OUTPUT_DIR_TRAIN):
    os.makedirs(OUTPUT_DIR_TRAIN, exist_ok=True)

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

for cam_id, sequence_name in enumerate(os.listdir(BASE_DIR)):
    seq_path = os.path.join(BASE_DIR, sequence_name)
    output_dir = os.path.join(OUTPUT_DIR_TRAIN, sequence_name)
    output_txt_ann_file = "train_list.txt" if sequence_name in TRAIN_SEQUENCES else "test_list.txt"
    output_txt_ann_path = os.path.join(OUTPUT_DIR_ANN_TRAIN, output_txt_ann_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.isdir(seq_path):
        continue
        
    gt = os.path.join(seq_path, 'gt', 'gt.txt')
    if not os.path.exists(gt): continue
    img_folder = os.path.join(BASE_DIR, sequence_name, "img1")

    with open(gt, "r") as gt_file:
        gt_data = gt_file.readlines()

    for ground_truth in tqdm(gt_data):
        frame_id, target_id, x, y, w, h, _, cls, *_ = ground_truth.strip().split(',')
        x, y, w, h = int(x), int(y), int(w), int(h)
        global_id = global_id_dict[(sequence_name, target_id)]
        img_name = f"{int(frame_id):07}.jpg"
        img_path = os.path.join(img_folder, img_name)
        filename = f"{global_id:4d}_c{cam_id}_f{int(frame_id):04d}.jpg"
        output_path = os.path.join(output_dir, filename)
        process_bbox_jobs.append((img_path, x, y, w, h, PADDING_PERCENTAGE, output_path))
        with open(output_txt_ann_path, 'a') as f:
            f.write(f"{filename} {global_id} {sequence_name}\n")

if PROCESS_IMAGES:
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_bbox, process_bbox_jobs), total=len(process_bbox_jobs)))

