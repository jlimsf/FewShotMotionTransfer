from PIL import Image
import glob
import numpy as np
from os.path import join as osp
import argparse
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument("root")
args = parser.parse_args()

# folders = glob.glob(osp(args.root, "*"))

folders = []
for mode in os.listdir(args.root):
    mode_dir = os.path.join(args.root, mode)
    for video in os.listdir(mode_dir):
        video_dir = os.path.join(mode_dir, video)
        for subject in os.listdir(video_dir):
            subject_dir = os.path.join(video_dir, subject)
            folders.append(subject_dir)


for folder in tqdm(folders):
    with open(osp(folder, "image_list.txt")) as f:
        files = f.readlines()

    backgrounds = []
    masks = []
    for file in files:
        try:

            image = np.asarray(Image.open(osp(folder, "image/{}.png".format(file.strip()))).convert("RGB")).astype(np.float32) / 255
            mask = Image.open(osp(folder, "segmentation/{}.png".format(file.strip()))).convert("RGB")
            mask = np.asarray(mask).astype(np.float32) / 255

            background = (1 - mask) * image
            backgrounds.append(background)
            masks.append(1 - mask)
        except Exception as e:
            print (e)
            continue


    b = backgrounds[0]
    m = masks[0]
    n = len(masks) // 5
    for i in range(1, len(masks)):
        index = np.logical_and(m < 0.5, masks[i] > 0.5)
        b[index] = backgrounds[i][index]
        m = np.clip(masks[i] + m, 0, 1)

    b = Image.fromarray((b * 255).astype(np.uint8))
    b.save(osp(folder, "background.png"))
