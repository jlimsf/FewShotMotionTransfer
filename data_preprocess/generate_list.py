import os
from os.path import join as osp
import glob
import argparse

ct = 0

def generate_list(root_path):
    files = glob.glob(osp(root_path, "image", "*.png"))
    with open(osp(root_path, "image_list.txt"), "w") as f:

        for file in files:

            name = os.path.basename(file).replace(".png", "")

            if os.path.exists(osp(root_path, "segmentation", name + ".png")) and \
               os.path.exists(osp(root_path, "body", name + ".png")) and \
               os.path.exists(osp(root_path, "densepose", name + ".png")):
                f.write(name + "\n")
                print ("Good")
            else:
                global ct
                ct += 1
                print (ct)
                continue


parser = argparse.ArgumentParser()
parser.add_argument("root")
args = parser.parse_args()

# root_paths = glob.glob(osp(args.root, "*"))

folders = []
for mode in os.listdir(args.root):
    mode_dir = os.path.join(args.root, mode)
    for video in os.listdir(mode_dir):
        video_dir = os.path.join(mode_dir, video)
        for subject in os.listdir(video_dir):
            subject_dir = os.path.join(video_dir, subject)
            folders.append(subject_dir)

for root_path in folders:

    generate_list(root_path)
