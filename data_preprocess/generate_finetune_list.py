import os
from os.path import join as osp
import glob
import argparse


def generate_list(root_path):
    files = glob.glob(osp(root_path, "image", "*.png"))

    with open(osp(root_path, "image_list.txt"), "w") as f:
        for file in files:
            name = os.path.basename(file).replace(".png", "")
            
            if os.path.exists(osp(root_path, "texture", name + ".png")) and \
               os.path.exists(osp(root_path, "segmentation", name + ".png")) and \
               os.path.exists(osp(root_path, "body", name + ".png")) and \
               os.path.exists(osp(root_path, "densepose", name + "_IUV.png")):
                f.write(name + "\n")


parser = argparse.ArgumentParser()
parser.add_argument("root")
parser.add_argument("--n_finetune", type=int, default=20)

args = parser.parse_args()

root_paths = glob.glob(osp(args.root, "*"))

for root_path in root_paths:

    with open(osp(root_path, "image_list.txt")) as f:
        lines = f.readlines()


    lines = lines[:args.n_finetune]
    with open(osp(root_path, "finetune_samples.txt"), "w") as f:
        f.writelines(lines)
