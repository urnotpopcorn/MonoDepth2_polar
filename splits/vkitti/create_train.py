import os
import sys
import numpy as np

input_dir = '/mnt/sdc/dataset/VKitti'
# output_file = 'train_files_woweather_ori.txt'
output_file = 'train_files_wweather_ori.txt'
total_file_list = list()

scene_list = os.listdir(input_dir)
case_list = ['15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right', 'fog', 'morning', 'overcast', 'rain', 'sunset']
# case_list = ['15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right'] 

with open(output_file, 'w') as f:
    for scene in scene_list:
        if 'tar' in scene:
            continue
        for case in case_list:
            sub_dir = os.path.join(input_dir, scene, case, 'frames/rgb/Camera_0')
            input_file_list = os.listdir(sub_dir)
            
            for input_file in input_file_list:
                input_idx = int(input_file.split('.')[0].split('_')[1])
                if input_idx == 0 or input_idx == len(input_file_list)-1:
                    continue

                sub_path = scene+'\t'+case+'\t'+input_file+'\n'
                f.write(sub_path)
                # total_file_list.append(sub_path)
