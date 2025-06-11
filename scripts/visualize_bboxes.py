import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

base_path = "/home/markoharalovic/VisDrone2019-MOT-test-dev/sequences"
output_dir = "vis"
os.makedirs(output_dir, exist_ok=True)

def generate_id_color_map(object_ids):
    id_list = sorted(list(object_ids))
    colormap = plt.get_cmap('gist_rainbow', len(id_list))  
    return {
        obj_id: tuple(int(c) for c in (np.array(colormap(i)[:3]) * 255))
        for i, obj_id in enumerate(id_list)
    }

for sequence_name in os.listdir(base_path):
    sequence_path = os.path.join(base_path, sequence_name)
    img_dir = os.path.join(sequence_path, "img1")
    gt_path = os.path.join(sequence_path, "gt", "gt.txt")

    if not os.path.exists(img_dir) or not os.path.exists(gt_path):
        continue

    print(f"Processing {sequence_name}...")

    gt = {}
    object_ids = set()
    with open(gt_path, 'r') as f:
        for line in f:
            fields = list(map(int, line.strip().split(',')[:6]))  # frame_id, obj_id, x, y, w, h
            frame_id, obj_id, x, y, w, h = fields
            object_ids.add(obj_id)
            if frame_id not in gt:
                gt[frame_id] = []
            gt[frame_id].append((obj_id, x, y, w, h))

    id_to_color = generate_id_color_map(object_ids)

    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    selected_images = random.sample(image_files, 5)

    sequence_vis_dir = os.path.join(output_dir, sequence_name)
    os.makedirs(sequence_vis_dir, exist_ok=True)

    for filename in selected_images:
        frame_id = int(filename.split('.')[0])
        img_path = os.path.join(img_dir, filename)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Failed to load {img_path}")
            continue

        for (obj_id, x, y, w, h) in gt.get(frame_id, []):
            color = id_to_color.get(obj_id, (255, 255, 255))
            color = tuple(int(c) for c in color)  
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, str(obj_id), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 6))
        plt.imshow(image_rgb)
        plt.title(f"{sequence_name} | {filename}")
        plt.axis('off')
        output_path = os.path.join(sequence_vis_dir, filename)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
