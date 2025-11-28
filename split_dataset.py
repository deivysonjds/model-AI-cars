import os
import random
import shutil

# PASTAS
IMAGES_DIR = "dataset/images"
LABELS_DIR = "dataset/labels"

OUTPUT_DIR = "dataset_yolo"
TRAIN = 0.7
VAL = 0.2
TEST = 0.1

os.makedirs(f"{OUTPUT_DIR}/train/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/train/labels", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/val/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/val/labels", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/test/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/test/labels", exist_ok=True)

images = [f for f in os.listdir(IMAGES_DIR) if f.endswith((".jpg", ".png"))]
random.shuffle(images)

total = len(images)
train_split = int(total * TRAIN)
val_split = int(total * VAL)

train_files = images[:train_split]
val_files = images[train_split:train_split + val_split]
test_files = images[train_split + val_split:]

def copy_files(file_list, folder):
    for img in file_list:
        label = img.replace(".jpg", ".txt").replace(".png", ".txt")

        shutil.copy(f"{IMAGES_DIR}/{img}", f"{OUTPUT_DIR}/{folder}/images/{img}")
        shutil.copy(f"{LABELS_DIR}/{label}", f"{OUTPUT_DIR}/{folder}/labels/{label}")

copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("Dataset separado com sucesso!")
