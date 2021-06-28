import os
import sys

input_file = "train_files.txt.ori"
delete_list = list()
with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()
        line = line.split()
       
        cnt = 0
        for idx in range(len(line)):
            input_path = os.path.join('/home/xzwu/xzwu/Code/MonoDepth2_stage1/dataset/', line[idx])
            if os.path.exists(input_path) == False:
                '''
                delete_path = line[idx].split('/')[1]
                if delete_path not in delete_list:
                    print(delete_path)
                    delete_list.append(delete_path)
                '''
                pass
            else:
                cnt+=1
                print(line[idx], end=' ')
        
        if cnt != 0:
            print()
'''
for delete_path in delete_list:
    print(delete_path)
'''
