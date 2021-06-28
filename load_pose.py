import sys
import numpy as np

def loadPoses(file_name):
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    file_len = len(s)
    poses = []
    for cnt, line in enumerate(s):
        P = np.eye(4)
        
        line_split = [float(i) for i in line.split()]
        withIdx = int(len(line_split) == 13)
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row*4 + col + withIdx]
        
        poses.append(P)
    
    return poses
    

def compute_abs_pose(filename):
    poses = loadPoses(filename)
    abs_poses = list()
    abs_pose = np.eye(4)
    abs_poses.append(abs_pose)

    for pose in poses:
        abs_pose = abs_pose @ np.linalg.inv(pose)
        abs_poses.append(abs_pose)
    
    return abs_poses

def compute_abs_pose_previous(filename):
    poses = loadPoses(filename)
    abs_poses = list()
    abs_pose = np.eye(4)
    abs_poses.append(abs_pose)

    for pose in poses:
        abs_pose = pose @ abs_pose
        abs_poses.append(abs_pose)
    
    return abs_poses

def main():
    filename = sys.argv[1]
    abs_poses = compute_abs_pose(filename)
    for abs_pose in abs_poses:
        for i in range(3):
            for j in range(4):
                print(abs_pose[i][j], end='\t')
        print()

main()
