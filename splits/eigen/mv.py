import os
import sys
import numpy as np
from PIL import Image  # using pillow-simd for increased speed

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_image_path(folder, frame_index, side, img_ext='.png'):
    data_path = '/home/xzwu/xzwu/Code/MonoDepth2_stage1/dataset/raw_data/'
    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    f_str = "{:010d}{}".format(frame_index, img_ext)
    image_path = os.path.join(
        data_path, folder, "image_0{}/data".format(side_map[side]), f_str)
    return image_path

#2011_09_26/2011_09_26_drive_0009_sync 0000000388 l
with open('test_files.txt', 'r') as f:
    for line in f:
        line = line.strip()
        line = line.split()
        folder, frame_index, side = line
        color_path = get_image_path(folder, int(frame_index), side)
        new_color_path = color_path.replace('raw_data', 'raw_data_test')
        new_color_dir = os.path.dirname(new_color_path)
        if os.path.exists(new_color_dir) == False:
            os.makedirs(new_color_dir)

        cmd = 'cp '+color_path+' '+new_color_path
        os.system(cmd)
        # print(cmd)
        # color = pil_loader(color_path)
        
        # print(color.size)
        

