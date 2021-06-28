import os
import cv2
import numpy as np

def find_correspondence_points(img1, img2, vis=False):
    sift = cv2.xfeatures2d.SIFT_create(400) # cv2.xfeatures2d.SIFT_create(400)
    # sift.setHessianThreshold(50000)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(
        cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
        
    kp2, des2 = sift.detectAndCompute(
        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
    
    if des1 is None or des2 is None or des1.shape[0] < 4 or des2.shape[0] < 4:
        return None, None, None
    
    if vis == True:
        img1_circle = cv2.drawKeypoints(img1, kp1, None, (255,0,0), 4)
        img2_circle = cv2.drawKeypoints(img2, kp2, None, (255,0,0), 4)

        cv2.imwrite('img1.jpg', img1_circle)
        cv2.imwrite('img2.jpg', img2_circle)

    # Find point matches
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, table_number=20, key_size=10, multi_probe_level=2)
    '''
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 6) # 12 key_size = 12, # 20 multi_probe_level = 1
    '''
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    des1 = np.array(des1).astype(np.float32)
    des2 = np.array(des2).astype(np.float32)

    matches = flann.knnMatch(des1, des2, k=2)
    matchesMask = [[0,0] for i in range(len(matches))]

    # Apply Lowe's SIFT matching ratio test
    good = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good.append(m)
            matchesMask[i]=[1,0]

         
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)
    # print(len(good))
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    if vis == True:
        cv2.imwrite('img3.jpg', img3)

    src_pts = np.asarray([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.asarray([kp2[m.trainIdx].pt for m in good])
    
    if len(src_pts) < 4:
        return None, None, None

    # Constrain matches to fit homography
    retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
    mask = mask.ravel()

    # We select only inlier points
    pts1 = src_pts[mask == 1]
    pts2 = dst_pts[mask == 1]
    # print(pts2.shape) # N, 2

    return pts1.T, pts2.T, img3

if __name__ == '__main__':
    img1 = cv2.imread('dataset/NYUv2/NYUv2_labeled/processed_data/rgb/00191.jpg')
    img2 = cv2.imread('dataset/NYUv2/NYUv2_labeled/processed_data/rgb/00192.jpg')
    find_correspondence_points(img1, img2)