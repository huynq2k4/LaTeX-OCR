import os
import gc
import glob
import yaml
import shutil

from ultralytics import YOLO

import cv2
import torch
import numpy as np
from tqdm import tqdm


def reformat_dataset(raw_data="/raw_data", images_root="/process/image", labels_root="/process/labels", image_size=(2016, 2016)):
    """_summary_

    Args:
        raw_data (_type_): path to the raw data
        images_root (_type_): path to the root directory where images will be copied
        labels_root (_type_): path to the root directory where labels will be copied
        image_size (tuple, optional): Defaults to (2016, 2016).
    """

    for subset in ['train', 'test', 'valid']:
        # Retrieving all images and labels for each subset of the data
        images = sorted(glob.glob(os.path.join(
            raw_data, 'images', subset, '*.jpg')))
        labels = sorted(glob.glob(os.path.join(
            raw_data, 'labels', subset, '*.txt')))

        for image, label in tqdm(zip(images, labels), total=len(images), desc=f'Copying {subset} data'):
            # Image copying
            shutil.copy(image, os.path.join(images_root, subset))

            # Label copying (have to process bounding box coordinates
            # since they are not yet normalized)
            label_name = label.split("/")[-1]
            with open(label, 'r') as lf:
                label_lines = map(lambda l: l.split(), lf.readlines())

            for line in label_lines:
                line = np.array(line, dtype=np.float16)

                y_min, x_min = line[[1, 2]]
                height, width = line[[3, 4]]

                x_center = str((x_min + width / 2) / image_size[1])
                y_center = str((y_min + height / 2) / image_size[0])
                width = str(width / image_size[1])
                height = str(height / image_size[0])

                with open(os.path.join(labels_root, subset, label_name), "a+") as lf:
                    lf.write(str(int(line[0])) + " ")
                    lf.write(" ".join([x_center, y_center, width, height]))
                    lf.write("\n")


def train_yolo():
    RANDOM_STATE = 42
    INPUT_SIZE = 1024
    N_EPOCHS = 5
    PATIENCE = 5
    BATCH_SIZE = 8
    CACHE_DATA = True
    DEVICES = 1

    torch.cuda.empty_cache()
    gc.collect()
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    model = YOLO('yolov8m.pt')
    model.train(
        data="data.yaml",
        epochs=N_EPOCHS,
        patience=PATIENCE,
        imgsz=INPUT_SIZE,
        batch=BATCH_SIZE,
        seed=RANDOM_STATE,
        cache=CACHE_DATA,
        device=DEVICES,
        project='Formula-Detection',
    )
    model.save('best.pt')
    
def evaluate_yolo(images=None, predict_root=None):
    best_weights = "best.pt"
    best_model = YOLO(best_weights) 
    metrics = best_model.val() 
    print(f"Mean Average Precision @.5:.95: {metrics.box.map}")    
    print(f"Mean Average Precision @.5:     {metrics.box.map50}") 
    print(f"Mean Average Precision @.7:     {metrics.box.map75}")
    
    if images is not None:
        with torch.no_grad():
            predictions = best_model.predict(
                source=images,
                conf=0.5,
                iou=0.2,
                stream=True
            )

        for prediction in predictions:
            if len(prediction.boxes.xyxy):
                name = prediction.path.split("/")[-1].split(".")[0]
                boxes = prediction.boxes.xyxy.cpu().numpy()
                scores = prediction.boxes.conf.cpu().numpy()
                
                label_path = os.path.join(predict_root, name + ".txt")
                
                with open(label_path, "w+") as f:
                    for score, box in zip(scores, boxes):
                        text = f"{score:0.4f} {' '.join(box.astype(str))}"
                        f.write(text)
                        f.write("\n")

if __name__ == "__main__":
    with open("config.yml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    reformat_dataset()

    train_yolo()

    evaluate_yolo()