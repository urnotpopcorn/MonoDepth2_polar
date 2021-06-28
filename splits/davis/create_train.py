import os
import sys

curclass = 'bmx-bumps'
num = 88
output_file = 'train_files.txt'
with open(output_file, 'w') as f:
    for i in range(num):
        string = curclass+'\t'+str(i+1).zfill(5)+'\n'
        # print(string)
        f.write(string)

