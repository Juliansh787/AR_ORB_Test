import numpy as np
import cv2
from matplotlib import pyplot as plt

## ORB 기술자
def ORBeaxmple():
    img1 = cv2.imread('harleyQuinnA.jpg',0)
    img2 = cv2.imread('harleyQuinnB.jpg',0)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=2)
    plt.imshow(img3), plt.show()

    return len(matches)

## SIFT 기술자
def SIFTexample():
    img1 = cv2.imread('harleyQuinnA.jpg',0)
    img2 = cv2.imread('harleyQuinnB.jpg',0)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.3*n.distance:
            good.append([m])
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    plt.imshow(img3),plt.show()

    return len(matches)

## FLANN은 Fast Library for Approximate Nearest Neighbors의 약자입니다. 대용량의 데이터셋과 고차원 특징점에 있어서 속도면에 최적화 되어 있음
def FLANNexample():
    img1 = cv2.imread('harleyQuinnA.jpg',0)
    img2 = cv2.imread('harleyQuinnB.jpg',0)

    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.3*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    plt.imshow(img3,),plt.show()

    return len(matches)

matches = FLANNexample()
print(matches)